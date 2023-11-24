import os

import numpy as np
from PIL import ImageDraw, Image

import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.transforms import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from models.generate.mgd_pipe import MGDPipe
from models.generate.mgd import mgd
from third_party import M2FPBatchInfer, DWPoseBatchInfer, PiDiNetBatchInfer
from tools import kpoint_to_heatmap, get_coco_palette


class MGDBatchInfer(object):
    def __init__(self,
                 infer_height: int = 512,
                 infer_width: int = 384,
                 device: str = "cuda:0",
                 unet_in_channels: int = 28,
                 unet_weight_path: str = None,
                 ):

        self.infer_height = infer_height
        self.infer_width = infer_width
        self.device = device

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        ''' 3rd party models '''
        self.parse_infer = None
        self.pose_infer = None
        self.edge_infer = None

        ''' mgd models '''
        self.unet_in_channels = unet_in_channels
        self.unet_weight_path = unet_weight_path
        self.unet = None
        self.text_encoder = None
        self.vae = None
        self.tokenizer = None
        self.val_scheduler = None
        self.mgd_pipe = None

    @torch.inference_mode()
    def forward_rgb_as_pil(self,
                           model_rgb: np.ndarray,
                           prompt: str,
                           warped_rgb: np.ndarray = None,
                           num_samples: int = 4,
                           seed: int = 42,
                           ):
        height, width = self.infer_height, self.infer_width
        device = self.device

        ''' parse '''
        if self.parse_infer is None:
            self.parse_infer = M2FPBatchInfer()
        parse_pil = self.parse_infer.forward_rgb_as_pil(model_rgb).resize((width, height), resample=Image.NEAREST)
        parse_pil.putpalette(get_coco_palette())

        ''' pose '''
        if self.pose_infer is None:
            self.pose_infer = DWPoseBatchInfer()
        pose_rgb = self.pose_infer.forward_rgb_as_rgb(model_rgb)
        pose_pil = Image.fromarray(pose_rgb.astype(np.uint8))
        pose_keypoint_dict = self.pose_infer.get_latest_keypoint_dict()

        ''' edge '''
        if self.edge_infer is None:
            self.edge_infer = PiDiNetBatchInfer()
        edge_ori_pil = self.edge_infer.forward_rgb_as_pil(model_rgb)
        # edge_ori_pil = Image.open("./tmp_ori_edge_modified.png")
        edge_ori_pil = Image.fromarray(np.array(edge_ori_pil).squeeze()).convert("L")
        edge_ori_pil = edge_ori_pil.resize((width, height), resample=Image.NEAREST)

        ''' warped '''
        if warped_rgb is not None:
            warped_pil = Image.fromarray(warped_rgb.astype(np.uint8))
            warped_down_rgb = np.array(warped_pil.resize((width, height), resample=Image.BILINEAR))

        ''' process input '''
        parse_dict = self._process_parse(parse_pil)
        pose_dict = self._process_pose(pose_keypoint_dict)
        edge_arr = np.array(edge_ori_pil) * parse_dict["cloth"]  # in [0,255]
        edge_pil = Image.fromarray(edge_arr.astype(np.uint8)).resize((width, height), resample=Image.NEAREST)
        edge_tensor = (torch.from_numpy(edge_arr).float() / 255.).unsqueeze(0)  # (1,H,W), in [0,1]
        inpaint_mask = self._process_inpaint_mask(
            pose_data=pose_dict["data"],
            parse_arms=parse_dict["arms"],
            parse_head=parse_dict["head"],
            parse_cloth=parse_dict["cloth"],
            parse_mask_fixed=parse_dict["fixed"],
            parse_mask_changeable=parse_dict["changeable"],
        )

        ''' vis inputs '''
        model_pil = Image.fromarray(model_rgb.astype(np.uint8))
        model_down_rgb = np.array(model_pil.resize((width, height), resample=Image.BILINEAR))
        # Image.fromarray(model_down_rgb).save("tmp_01_model.png")
        # edge_pil.save("tmp_02_edge.png")
        inpaint_mask_arr = (inpaint_mask[0]).cpu().numpy().astype(np.uint8)  # (H,W), in [0,1]
        # Image.fromarray(inpaint_mask_arr * 255).save("tmp_03_inpaint_mask.png")
        model_agnostic_rgb = model_down_rgb * (1 - inpaint_mask_arr[:, :, np.newaxis])
        # Image.fromarray(model_agnostic_rgb).save("tmp_04_model_agnostic.png")

        ''' load mgd pipeline '''
        if self.mgd_pipe is None:
            self.mgd_pipe = self._load_mgd_models_and_pipeline()

        ''' prepare inputs for DDIM '''
        guidance_scale = 7.5
        prompts = [
            prompt,
        ]
        """ e.g.
        flared sleeves white blouse button white mandarin collar
        the camel colored shirt is made from a cotton blend
        """

        model_input = self.trans(model_down_rgb).unsqueeze(0).cuda()
        mask_input = inpaint_mask.unsqueeze(0).cuda()
        sketch_input = edge_tensor.unsqueeze(0).cuda()
        pose_map_input = pose_dict["map"].unsqueeze(0).cuda()
        warped_input = self.trans(warped_down_rgb).unsqueeze(0).cuda() if warped_rgb is not None else None
        print("tensors:", model_input.shape, mask_input.shape, sketch_input.shape, pose_map_input.shape)

        generator = torch.Generator("cuda").manual_seed(seed)
        sketch_cond_rate = 0.2
        start_cond_rate = 0.0
        no_pose = False

        neg_prompts = [""]

        ''' run '''
        generated_images = self.mgd_pipe(
            prompt=prompts,
            image=model_input,
            mask_image=mask_input,
            pose_map=pose_map_input,
            warped=warped_input,
            sketch=sketch_input,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            sketch_cond_rate=sketch_cond_rate,
            start_cond_rate=start_cond_rate,
            no_pose=no_pose,
            negative_prompt=neg_prompts,
        ).images  # num_samples*[Image.Image]

        ''' vis outputs '''
        print("model_input:", model_input[0].min(), model_input[0].max())  # in [-1,1]
        ret_pils = []
        model_i0 = model_input[0] * 0.5 + 0.5
        parse_i0 = parse_dict["tensor"][0].to(device)
        for i in range(len(generated_images)):
            generated_images[i].save(f"tmp_y_gen_{seed}_{i:02d}.png")
            final_img = self._compose_img(model_i0, generated_images[i], parse_i0)
            final_img = transforms.ToPILImage()(final_img)
            final_img.save(os.path.join(".", f"tmp_z_result_{i:02d}.png"))
            ret_pils.append(final_img)
        return ret_pils

    def _process_parse(self, parse_pil: Image.Image) -> dict:
        parse_array = np.array(parse_pil)
        parse_tensor = torch.from_numpy(parse_array).long().unsqueeze(0).unsqueeze(0)  # (1,1,H,W), in {0,...,#seg}

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)

        parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                            (parse_array == 2).astype(np.float32) + \
                            (parse_array == 18).astype(np.float32) + \
                            (parse_array == 19).astype(np.float32)

        # parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
        parser_mask_changeable = (parse_array == 0).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)
        parse_mask = (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)

        parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32)  # the lower body is fixed

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        return {
            "tensor": parse_tensor,
            "cloth": parse_mask,
            "arms": arms,
            "head": parse_head,
            "fixed": parser_mask_fixed,
            "changeable": parser_mask_changeable,
        }

    def _process_pose(self, pose_keypoint_dict: dict,
                      point_num: int = 18, radius: int = 5,
                      ) -> dict:
        pose_data = pose_keypoint_dict['people'][0]['person_keypoints_2d']['candidate']
        print(type(pose_data), pose_data.shape, pose_data.min(), pose_data.max())  # in [0,1]
        # pose_data = np.array(pose_data)
        # pose_data = pose_data.reshape((-1, 3))[:, :2]  # for dwpose, no need to reshape

        # rescale keypoints on the base of height and width
        # pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
        # pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

        height, width = self.infer_height, self.infer_width

        pose_data[:, 0] = pose_data[:, 0] * width
        pose_data[:, 1] = pose_data[:, 1] * height

        pose_map = torch.zeros(point_num, height, width)
        r = radius * (height / 512.0)
        im_pose = Image.new('L', (width, height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (width, height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (width, height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], 1)
            point_y = np.multiply(pose_data[i, 1], 1)

            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                      'white')
            one_map = self.trans(one_map)
            pose_map[i] = one_map[0]

        d = []

        for idx in range(point_num):
            ux = pose_data[idx, 0]  # / (192)
            uy = (pose_data[idx, 1])  # / (256)

            # scale posemap points
            px = ux  # * self.width
            py = uy  # * self.height

            d.append(kpoint_to_heatmap(np.array([px, py]), (height, width), 9))

        pose_map = torch.stack(d)  # got it, (18,H,W)
        return {
            "data": pose_data,
            "map": pose_map,
        }

    def _process_inpaint_mask(self, pose_data: np.ndarray,
                              parse_arms: torch.Tensor,
                              parse_head: torch.Tensor,
                              parse_cloth: torch.Tensor,
                              parse_mask_fixed: torch.Tensor,
                              parse_mask_changeable: torch.Tensor,
                              ) -> torch.Tensor:
        width, height = self.infer_width, self.infer_height

        im_arms = Image.new('L', (width, height))
        arms_draw = ImageDraw.Draw(im_arms)

        # do in any case because i have only upperbody
        data = pose_data

        # rescale keypoints on the base of height and width
        # data[:, 0] = data[:, 0] * (self.width / 768)
        # data[:, 1] = data[:, 1] * (self.height / 1024)

        shoulder_right = np.multiply(tuple(data[2]), 1)
        shoulder_left = np.multiply(tuple(data[5]), 1)
        elbow_right = np.multiply(tuple(data[3]), 1)
        elbow_left = np.multiply(tuple(data[6]), 1)
        wrist_right = np.multiply(tuple(data[4]), 1)
        wrist_left = np.multiply(tuple(data[7]), 1)

        ARM_LINE_WIDTH = int(90 / 512 * height)
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line(
                    np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line(
                    np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

        hands = np.logical_and(np.logical_not(im_arms), parse_arms)
        parse_cloth += im_arms
        parse_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)

        parse_mask_fixed = np.logical_or(parse_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_cloth += np.logical_or(parse_cloth, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                               np.logical_not(
                                                                   np.array(parse_head_2, dtype=np.uint16))))

        parse_mask = np.logical_and(parse_mask_changeable, np.logical_not(parse_cloth))
        parse_mask_total = np.logical_or(parse_mask, parse_mask_fixed)
        # im_mask = image * parse_mask_total
        inpaint_mask = 1 - parse_mask_total

        # here we have to modify the mask and get the bounding box
        bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
        bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
        xmin = bboxes[0, 0]
        xmax = bboxes[0, 2]
        ymin = bboxes[0, 1]
        ymax = bboxes[0, 3]

        inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
            torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
            torch.logical_not(parse_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

        inpaint_mask = inpaint_mask.unsqueeze(0)  # (1,H,W), in [0,1]
        print("inpaint_mask:", inpaint_mask.shape)
        return inpaint_mask

    def _load_mgd_models_and_pipeline(self):
        if self.unet is None:
            self.unet = mgd(
                pretrained=True,
                in_channels=self.unet_in_channels,
                weight_path=self.unet_weight_path,
            )
        sd_inpaint_dir = "pretrained/stable-diffusion-inpainting"
        if self.text_encoder is None:
            self.text_encoder = CLIPTextModel.from_pretrained(sd_inpaint_dir + "/text_encoder")
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(sd_inpaint_dir + "/vae")
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(sd_inpaint_dir + "/tokenizer")
        if self.val_scheduler is None:
            self.val_scheduler = DDIMScheduler.from_pretrained(sd_inpaint_dir + "/scheduler")
            self.val_scheduler.set_timesteps(50)

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

        # Enable memory efficient attention if requested
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Move text_encode and vae to gpu and cast to weight_dtype
        device = "cuda:0"
        weight_dtype = torch.float32
        self.text_encoder.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.unet.to(device)
        self.unet.eval()

        # Select fast classifier free guidance
        mgd_pipe = MGDPipe(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.val_scheduler,
        )
        mgd_pipe.enable_attention_slicing()
        return mgd_pipe

    @staticmethod
    def _compose_img(gt_img, fake_img, im_parse):
        seg_head = torch.logical_or(im_parse == 1, im_parse == 2)
        seg_head = torch.logical_or(seg_head, im_parse == 4)
        seg_head = torch.logical_or(seg_head, im_parse == 13)

        true_head = gt_img * seg_head
        true_parts = true_head

        generated_body = (transforms.ToTensor()(fake_img).cuda()) * (~(seg_head))
        return true_parts + generated_body

    @staticmethod
    def _get_viton_labels():
        seg_viton_labels = {
            0: "background",
            1: "hat",
            2: "hair",
            3: "glove",
            4: "sunglasses",
            5: "upper-clothes",
            6: "dress",
            7: "coat",
            8: "socks",
            9: "pants",
            10: "torso-skin",  # in lip, it is jumpsuit
            11: "scarf",
            12: "skirt",
            13: "face",
            14: "left-arm",
            15: "right-arm",
            16: "left-leg",
            17: "right-leg",
            18: "left-shoe",
            19: "right-shoe"
        }
        return seg_viton_labels
