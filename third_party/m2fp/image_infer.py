try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore")
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set
import PIL.Image
import numpy as np
from einops import rearrange

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from m2fp import (
    SemanticSegmentorWithTTA,
    ParsingWithTTA,
    M2FPSemanticHPDatasetMapper,
    M2FPParsingDatasetMapper,
    M2FPParsingLSJDatasetMapper,
    ParsingEvaluator,
    WandBWriter,
    add_m2fp_config,
    build_detection_test_loader,
    load_image_into_numpy_array,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # parsing
        if evaluator_type == "parsing":
            evaluator_list.append(
                ParsingEvaluator(
                    dataset_name,
                    cfg.MODEL.M2FP.TEST.PARSING_INS_SCORE_THR,
                    output_dir=output_folder,
                    parsing_metrics=cfg.MODEL.M2FP.TEST.METRICS
                )
            )
        # semantic segmentation
        if evaluator_type == "sem_seg":
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    sem_seg_loading_fn=load_image_into_numpy_array
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # single human parsing dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "m2fp_semantic_hp":
            mapper = M2FPSemanticHPDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # multiple human parsing dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "m2fp_parsing":
            mapper = M2FPParsingDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # multiple human parsing dataset mapper with lsj
        elif cfg.INPUT.DATASET_MAPPER_NAME == "m2fp_parsing_lsj":
            mapper = M2FPParsingLSJDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite func:`detectron2.data.build_detection_test_loader`,
        to adapt the single parsing test loader for lip and ATR, etc.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        if cfg.MODEL.M2FP.TEST.PARSING_ON:
            model = ParsingWithTTA(cfg, model)
        else:
            model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_writers(self):
        default_writers = super(Trainer, self).build_writers()
        if self.cfg.WANDB.ENABLED:
            default_writers.append(WandBWriter(self.cfg))
        return default_writers


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_m2fp_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = make_abs_path("../../pretrained/m2fp/cihp/model_final.pth")
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "m2fp" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def _save_as_pil(save_path : str, in_data):
    if isinstance(in_data, np.ndarray):
        pil = PIL.Image.fromarray(in_data)
    elif isinstance(in_data, torch.Tensor):
        pil = tensor_to_rgb(in_data, out_as_pil=True)
    elif isinstance(in_data, PIL.Image.Image):
        pil = in_data
    else:
        raise TypeError(f"Input type not supported: {type(in_data)}")
    pil.save(save_path)


def tensor_to_rgb(x: torch.Tensor,
                  out_batch_idx: int = 0,
                  out_as_pil: bool = False,
                  is_zero_center: bool = True,
                  ):
    ndim = x.ndim
    b = x.shape[0]
    if ndim == 4:  # (B,C,H,W), e.g. image
        x = rearrange(x, "b c h w -> b h w c").contiguous()
    elif ndim == 3:  # (B,H,W), e.g. mask
        x = x.unsqueeze(-1)
        x = torch.cat([x, x, x], dim=-1)  # (B,H,W,3)

    img = x.detach().cpu().numpy().astype(np.float32)  # (B,H,W,3)
    if is_zero_center:
        img = (img + 1.) * 127.5
    else:
        img = img * 255.

    def to_pil(in_x: np.ndarray, use_pil: bool):
        out_x = in_x.astype(np.uint8)
        if use_pil:
            out_x = PIL.Image.fromarray(out_x)
        return out_x

    if out_batch_idx is None:
        ret = [to_pil(img[i], out_as_pil) for i in range(b)]
    else:
        ret = to_pil(img[out_batch_idx], out_as_pil)

    return ret


class M2FPBatchInfer(object):
    def __init__(self):
        args = default_argument_parser().parse_args()
        args.config_file = make_abs_path("./configs/cihp/m2fp_R101_bs16_265k.yaml")
        args.eval_only = True

        cfg = setup(args)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()

        x_fn = "hoodie_cloth.jpg"
        x_in = np.array(PIL.Image.open(x_fn).convert("RGB"))
        x_in = torch.FloatTensor(x_in).permute(2, 0, 1)
        x_in = (x_in / 127.5) - 1.
        apply_input = [{"image": x_in, }]
        y_out = model(apply_input)[0]
        print(y_out.keys())
        print(y_out["parsing"].keys())
        for k, v in y_out["parsing"].items():
            print(k, ":", len(v), type(v))
            idx = 0
            for w in v:
                print(type(w))
                if isinstance(w, dict):
                    print(w["category_id"], w["score"], w["mask"].shape, w["mask"].min(), w["mask"].max())
                    mask = w["mask"]
                    mask[mask >= 0.5] = 1
                    mask[mask < 0.5] = 0
                    mask = mask * 2. - 1.
                    mask = mask.unsqueeze(0)
                    category_id = w["category_id"]
                    score = "%.2f" % w["score"]
                    save_fn = f"{x_fn}_c{category_id}_id{idx}.png"
                    _save_as_pil(save_fn, mask)
                elif isinstance(w, torch.Tensor):
                    s = w.detach()
                    print(s.shape, s.min(), s.max())
                    s[s >= 0.5] = 1
                    s[s < 0.5] = 0
                    s = s * 2. - 1.
                    s = s.unsqueeze(0)
                    _save_as_pil(f"semantic_{idx}.png", s)
                idx += 1

        # res = Trainer.test(cfg, model)
        #
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        # return res


if __name__ == "__main__":
    infer = M2FPBatchInfer()
