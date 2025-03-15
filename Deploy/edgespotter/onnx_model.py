from typing import Union
from pathlib import Path

import torch
from torch import nn
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from .adet.config import get_cfg


def setup_cfg(config_path: Union[str, Path], weights_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    weights = ["MODEL.WEIGHTS", weights_path]
    cfg.merge_from_list(weights)
    return cfg


def SimpleONNXReadyModel(config_path, weights_path,
                         width=960, height=540):
    cfg = setup_cfg(config_path, weights_path)
    cfg.freeze()
    return Predictor(cfg, width, height)
        

class Predictor(nn.Module):
    def __init__(self, cfg, input_width: int, 
                 input_height: int):
        super().__init__()
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        self.input_width = input_width
        self.input_height = input_height

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)


        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def forward(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        with torch.no_grad(): 
            inputs = {"image": original_image, "height": self.input_height, "width": self.input_width}
            predictions = self.model([inputs])
            return predictions