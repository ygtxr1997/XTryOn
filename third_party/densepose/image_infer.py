import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import cv2
import numpy as np
from PIL import Image
import torch

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))
sys.path.append(make_abs_path('./'))
from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
    DensePosePilVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)


class Detectron2BatchInfer(object):

    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePosePilVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    def __init__(self, args: argparse.Namespace = None):
        if args is None:
            args = argparse.Namespace()
        args.cfg = make_abs_path("./configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml")
        args.model = make_abs_path("../../pretrained/densepose/model_final_844d15.pkl")
        args.visualizations = "dp_segm"
        args.opts = []

        args.min_score = 0.8
        args.nms_thresh = None
        args.texture_atlas = None
        args.texture_atlases_map = None

        self.args = args
        opts = []
        self.cfg = self.setup_config(args.cfg, args.model, args, opts)
        self.predictor = DefaultPredictor(self.cfg)
        print(f"[Detectron2BatchInfer] Loading model from {args.model}")

    def forward_rgb_as_pil(self, x_arr: np.ndarray):
        context = self.create_context(self.args, self.cfg)
        x_arr = cv2.cvtColor(x_arr, cv2.COLOR_RGB2BGR)  # predictor expects BGR image
        with torch.no_grad():
            outputs = self.predictor(x_arr)["instances"]
            res_pil = self.execute_on_outputs(context, {"image": x_arr}, outputs)
        return res_pil

    def setup_config(self, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: list = None):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    def create_context(self, args: argparse.Namespace, cfg: CfgNode):
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(
        self, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        visualizer = context["visualizer"]
        extractor = context["extractor"]

        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)

        image_vis = visualizer.visualize(image, data)  # return PIL.Image with palette

        entry_idx = context["entry_idx"] + 1
        context["entry_idx"] += 1
        return image_vis


if __name__ == "__main__":
    test_args = argparse.Namespace()
    test_args.input = "tmp_00006_00.jpg"
    test_args.output = "image_densepose_contour.png"

    test_in_image = np.array(Image.open(test_args.input)).astype(np.uint8)
    image_infer = Detectron2BatchInfer()

    test_out = image_infer.forward_rgb_as_pil(test_in_image)
    test_out.save(test_args.output)
