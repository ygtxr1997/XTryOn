# Part of the implementation is borrowed and modified from M2FP, made publicly available
# under the CC BY-NC 4.0 License at https://github.com/soeaver/M2FP
import os
from typing import Any, Dict, List, Union
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from PIL import Image
import numpy as np
from easydict import EasyDict as edict

from .file_utils import func_receive_dict_inputs
from .logger import get_logger
from .backbone import build_resnet_deeplab_backbone
from .parsing_utils import center_to_target_size_test
from .m2fp.m2fp_decoder import MultiScaleMaskedTransformerDecoder
from .m2fp.m2fp_encoder import MSDeformAttnPixelDecoder
from .outputs import OutputKeys

logger = get_logger()


class ImageList(object):

    def __init__(self, tensor, image_sizes):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    def __getitem__(self, idx):
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., :size[0], :size[1]]

    @torch.jit.unused
    def to(self, *args, **kwargs):
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self):
        return self.tensor.device

    @staticmethod
    def from_tensors(tensors, size_divisibility=0, pad_value=0.0):
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [
                0, max_size[-1] - image_size[1], 0,
                max_size[-2] - image_size[0]
            ]
            batched_imgs = F.pad(
                tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(
                tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)


class M2FP(nn.Module):

    def __init__(self,
                 model_dir,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 pretrained=None,
                 input_single_human=None,
                 classes=None,
                 num_parsing=None,
                 single_human=True,
                 parsing_ins_score_thr=0.5,
                 parsing_on=False,
                 semantic_on=True,
                 sem_seg_postprocess_before_inference=True,
                 **kwargs):
        """
        Deep Learning Technique for Human Parsing: A Survey and Outlook. See https://arxiv.org/abs/2301.00394
        Args:
            backbone (dict): backbone config.
            encoder (dict): encoder config.
            decoder (dict): decoder config.
            pretrained (bool): whether to use pretrained model
            input_single_human (dict): input size config for single human parsing
            classes (list): class names
            num_parsing (int): total number of parsing instances, only for multiple human parsing
            single_human (bool): whether the task is single human parsing
            parsing_ins_score_thr: instance score threshold for multiple human parsing
            parsing_on (bool): whether to parse results, only for multiple human parsing
            semantic_on (bool): whether to output semantic map
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
        """
        super(M2FP, self).__init__(**kwargs)
        with open(os.path.join(model_dir, "configuration.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        f.close()

        model_cfg = cfg['model']
        backbone = model_cfg['backbone'] if backbone is None else backbone
        encoder = model_cfg['encoder'] if encoder is None else encoder
        decoder = model_cfg['decoder'] if decoder is None else decoder
        pretrained = model_cfg['pretrained'] if pretrained is None else pretrained
        if 'input_single_human' in model_cfg.keys():
            input_single_human = model_cfg['input_single_human'] if input_single_human is None else input_single_human
        classes = model_cfg['classes'] if classes is None else classes
        num_parsing = model_cfg['num_parsing'] if num_parsing is None else num_parsing

        single_human = model_cfg['single_human']
        parsing_ins_score_thr = model_cfg['parsing_ins_score_thr'] 
        parsing_on = model_cfg['parsing_on']
        semantic_on = model_cfg['semantic_on']
        sem_seg_postprocess_before_inference = model_cfg['sem_seg_postprocess_before_inference']

        self.model = edict(model_cfg)

        self.register_buffer(
            'pixel_mean',
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer(
            'pixel_std',
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        self.size_divisibility = 32

        self.backbone = build_resnet_deeplab_backbone(
            **backbone, input_shape={'channels': 3})
        in_features = encoder.pop('in_features')
        input_shape = {
            k: v
            for k, v in self.backbone.output_shape().items()
            if k in in_features
        }
        encoder = MSDeformAttnPixelDecoder(input_shape=input_shape, **encoder)
        decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=encoder.conv_dim, **decoder)
        self.sem_seg_head = M2FPHead(
            pixel_decoder=encoder, transformer_predictor=decoder)
        self.num_classes = decoder.num_classes
        self.num_queries = decoder.num_queries
        self.test_topk_per_image = 100

        self.input_single_human = input_single_human
        self.classes = classes
        self.num_parsing = num_parsing
        self.single_human = single_human
        self.parsing_ins_score_thr = parsing_ins_score_thr
        self.parsing_on = parsing_on
        self.semantic_on = semantic_on
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference or parsing_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        if pretrained:
            model_path = os.path.join(model_dir, "pytorch_model.pt")
            logger.info(f'loading model from {model_path}')
            weight = torch.load(model_path, map_location='cpu')['model']
            tgt_weight = self.state_dict()
            for name in list(weight.keys()):
                if name in tgt_weight:
                    load_size = weight[name].size()
                    tgt_size = tgt_weight[name].size()
                    mis_match = False
                    if len(load_size) != len(tgt_size):
                        mis_match = True
                    else:
                        for n1, n2 in zip(load_size, tgt_size):
                            if n1 != n2:
                                mis_match = True
                                break
                    if mis_match:
                        logger.info(
                            f'size mismatch for {name} '
                            f'({load_size} -> {tgt_size}), skip loading.')
                        del weight[name]
                else:
                    logger.info(
                        f'{name} doesn\'t exist in current model, skip loading.'
                    )

            self.load_state_dict(weight, strict=False)
            logger.info('load model done')

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        # Adapting a model with only one dict arg, and the arg name must be input or inputs
        if func_receive_dict_inputs(self.forward):
            return self.postprocess(self.forward(self.preprocess(args[0], **kwargs)))
        else:
            return self.postprocess(self.forward(self.preprocess(*args, **kwargs)))

    def _get_preprocess_shape(self, oldh, oldw, short_edge_length, max_size):
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def preprocess(self,
                   input: Image.Image,
                   min_size=640,
                   max_size=1333) -> Dict[str, Any]:
        image = input.convert("RGB")
        w, h = image.size[:2]
        dataset_dict = {'width': w, 'height': h}
        if self.model.single_human:
            image = np.asarray(image)
            image, crop_box = center_to_target_size_test(
                image, self.model.input_single_human['sizes'][0])
            dataset_dict['image'] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict['crop_box'] = crop_box
        else:
            new_h, new_w = self._get_preprocess_shape(h, w, min_size, max_size)
            test_transforms = T.Compose([
                T.Resize((new_h, new_w)),
                T.ToTensor(),
            ])
            image = test_transforms(image)
            dataset_dict['image'] = image * 255.
        result = {'batched_inputs': [dataset_dict]}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        batched_inputs = input['batched_inputs']
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        return dict(
            outputs=outputs, batched_inputs=batched_inputs, images=images)

    def postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        score_thr = 0.0
        outputs = input['outputs']
        batched_inputs = input['batched_inputs']
        images = input['images']
        if self.training:
            raise NotImplementedError
        else:
            mask_cls_results = outputs['pred_logits']  # (B, Q, C+1)
            mask_pred_results = outputs['pred_masks']  # (B, Q, H, W)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs,
                    images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                processed_results.append({})  # for each image

                if self.sem_seg_postprocess_before_inference:
                    if not self.single_human:
                        mask_pred_result = self.sem_seg_postprocess(
                            mask_pred_result, image_size, height, width)
                    else:
                        mask_pred_result = self.single_human_sem_seg_postprocess(
                            mask_pred_result, image_size,
                            input_per_image['crop_box'], height, width)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = self.semantic_inference(mask_cls_result,
                                                mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        if not self.single_human:
                            r = self.sem_seg_postprocess(
                                r, image_size, height, width)
                        else:
                            r = self.single_human_sem_seg_postprocess(
                                r, image_size, input_per_image['crop_box'],
                                height, width)
                        processed_results[-1]['sem_seg'] = r

                # parsing inference
                if self.parsing_on:
                    parsing_r = self.instance_parsing_inference(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]['parsing'] = parsing_r

            results = dict(eval_result=processed_results)

            predictions = results['eval_result'][0]
            class_names = self.classes
            results_dict = {
                OutputKeys.MASKS: [],
                OutputKeys.LABELS: [],
                OutputKeys.SCORES: []
            }
            if 'sem_seg' in predictions:
                semantic_pred = predictions['sem_seg']
                semantic_seg = semantic_pred.argmax(dim=0).detach().cpu().numpy()
                semantic_pred = semantic_pred.sigmoid().detach().cpu().numpy()
                class_ids = np.unique(semantic_seg)
                for class_id in class_ids:
                    label = class_names[class_id]
                    mask = np.array(semantic_seg == class_id, dtype=np.float64)
                    score = (mask * semantic_pred[class_id]).sum() / (
                        mask.sum() + 1)
                    results_dict[OutputKeys.SCORES].append(score)
                    results_dict[OutputKeys.LABELS].append(label)
                    results_dict[OutputKeys.MASKS].append(mask)
            elif 'parsing' in predictions:
                parsing_res = predictions['parsing']
                part_outputs = parsing_res['part_outputs']
                human_outputs = parsing_res['human_outputs']

                # process semantic_outputs
                for output in part_outputs + human_outputs:
                    score = output['score']
                    label = class_names[output['category_id']]
                    mask = (output['mask'] > 0).float().detach().cpu().numpy()
                    if score > score_thr:
                        results_dict[OutputKeys.SCORES].append(score)
                        results_dict[OutputKeys.LABELS].append(label)
                        results_dict[OutputKeys.MASKS].append(mask)
            else:
                raise NotImplementedError
            
            return results_dict
    
    @property
    def device(self):
        return self.pixel_mean.device

    def single_human_sem_seg_postprocess(self, result, img_size, crop_box,
                                         output_height, output_width):
        result = result[:, :img_size[0], :img_size[1]]
        result = result[:, crop_box[1]:crop_box[3],
                        crop_box[0]:crop_box[2]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False)[0]
        return result

    def sem_seg_postprocess(self, result, img_size, output_height,
                            output_width):
        result = result[:, :img_size[0], :img_size[1]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False)[0]
        return result

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(
            mask_cls, dim=-1)[..., :-1]  # discard non-sense category
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
        return semseg

    def instance_parsing_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(
            self.num_classes,
            device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                    1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        binary_pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * binary_pred_masks.flatten(1)).sum(1) / \
                                (binary_pred_masks.flatten(1).sum(1) + 1e-6)

        pred_scores = scores_per_image * mask_scores_per_image
        pred_labels = labels_per_image
        pred_masks = mask_pred

        # prepare outputs
        part_instance_res = []
        human_instance_res = []

        # bkg and part instances
        bkg_part_index = torch.where(pred_labels != self.num_parsing)[0]
        bkg_part_labels = pred_labels[bkg_part_index]
        bkg_part_scores = pred_scores[bkg_part_index]
        bkg_part_masks = pred_masks[bkg_part_index, :, :]

        # human instances
        human_index = torch.where(pred_labels == self.num_parsing)[0]
        human_labels = pred_labels[human_index]
        human_scores = pred_scores[human_index]
        human_masks = pred_masks[human_index, :, :]

        semantic_res = self.paste_instance_to_semseg_probs(
            bkg_part_labels, bkg_part_scores, bkg_part_masks)

        # part instances
        part_index = torch.where(bkg_part_labels != 0)[0]
        part_labels = bkg_part_labels[part_index]
        part_scores = bkg_part_scores[part_index]
        part_masks = bkg_part_masks[part_index, :, :]

        # part instance results
        for idx in range(part_labels.shape[0]):
            if part_scores[idx] < 0.1:
                continue
            part_instance_res.append({
                'category_id':
                part_labels[idx].cpu().tolist(),
                'score':
                part_scores[idx].cpu().tolist(),
                'mask':
                part_masks[idx],
            })

        # human instance results
        for human_idx in range(human_scores.shape[0]):
            if human_scores[human_idx] > 0.1:
                human_instance_res.append({
                    'category_id':
                    human_labels[human_idx].cpu().tolist(),
                    'score':
                    human_scores[human_idx].cpu().tolist(),
                    'mask':
                    human_masks[human_idx],
                })

        return {
            'semantic_outputs': semantic_res,
            'part_outputs': part_instance_res,
            'human_outputs': human_instance_res,
        }

    def paste_instance_to_semseg_probs(self, labels, scores, mask_probs):
        im_h, im_w = mask_probs.shape[-2:]
        semseg_im = []
        for cls_ind in range(self.num_parsing):
            cate_inds = torch.where(labels == cls_ind)[0]
            cate_scores = scores[cate_inds]
            cate_mask_probs = mask_probs[cate_inds, :, :].sigmoid()
            semseg_im.append(
                self.paste_category_probs(cate_scores, cate_mask_probs, im_h,
                                          im_w))

        return torch.stack(semseg_im, dim=0)

    def paste_category_probs(self, scores, mask_probs, h, w):
        category_probs = torch.zeros((h, w),
                                     dtype=torch.float32,
                                     device=mask_probs.device)
        paste_times = torch.zeros((h, w),
                                  dtype=torch.float32,
                                  device=mask_probs.device)

        index = scores.argsort()
        for k in range(len(index)):
            if scores[index[k]] < self.parsing_ins_score_thr:
                continue
            ins_mask_probs = mask_probs[index[k], :, :] * scores[index[k]]
            category_probs = torch.where(ins_mask_probs > 0.5,
                                         ins_mask_probs + category_probs,
                                         category_probs)
            paste_times += torch.where(ins_mask_probs > 0.5, 1, 0)

        paste_times = torch.where(paste_times == 0, paste_times + 1,
                                  paste_times)
        category_probs /= paste_times

        return category_probs


class M2FPHead(nn.Module):

    def __init__(self, pixel_decoder: nn.Module,
                 transformer_predictor: nn.Module):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions
