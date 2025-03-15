from typing import Optional
import numpy as np
import torch
from torch import nn
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances
from ..layers.pos_encoding import PositionalEncoding2D
from ..modeling.model.detection_transformer import DETECTION_TRANSFORMER
from ..utils.misc import NestedTensor


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images: ImageList) -> dict[torch.Tensor, torch.Tensor]:
        # print ("="*100)
        # print ("MaskedBackbone(images)")
        # print ("="*100)
        return self.backbone(images.tensor)


    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(
            results: list[Instances], 
            output_height: int, 
            output_width: int, 
            min_size: Optional[int] = None, 
            max_size: Optional[int] = None) -> list[Instances]:
    """
    scale align
    """
    if min_size and max_size:
        # to eliminate the padding influence for ViTAE backbone results
        size = min_size * 1.0
        scale_img_size = min_size / min(output_width, output_height)
        if output_height < output_width:
            newh, neww = size, scale_img_size * output_width
        else:
            newh, neww = scale_img_size * output_height, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        scale_x, scale_y = (output_width / neww, output_height / newh)
    else:
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    # scale points
    if results.has("ctrl_points"):
        ctrl_points = results.ctrl_points
        ctrl_points[:, 0::2] *= scale_x
        ctrl_points[:, 1::2] *= scale_y

    if results.has("bd") and not isinstance(results.bd, list):
        bd = results.bd
        bd[..., 0::2] *= scale_x
        bd[..., 1::2] *= scale_y

    return results


class TransformerPureDetector(nn.Module):
    """
        Same as :class:`detectron2.modeling.ProposalNetwork`.
        Use one stage detector and a second stage for instance-wise prediction.
        """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        # This is for ONNX export
        self.output_dims = {"height": cfg.INPUT.HEIGHT,
                            "weight": cfg.INPUT.WIDTH}
        
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(
            d2_backbone,
            PositionalEncoding2D(N_steps, cfg.MODEL.TRANSFORMER.TEMPERATURE, normalize=True)
        )
        backbone.num_channels = d2_backbone.num_channels
        self.detection_transformer = DETECTION_TRANSFORMER(cfg, backbone)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        
        # if `batched_inputs` contains tensors
        if type(batched_inputs[0]) is torch.Tensor:
            images = [self.normalizer(x.to(self.device)) for x in batched_inputs]
        else:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs) -> list[Instances]:
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """

        # if `batched_inputs` contains tensors - we are in an ONNX export
        # process, so we create a new input object
        batched_inputs_dict = batched_inputs
        if type(batched_inputs[0]) is torch.Tensor:
            batched_inputs_dict = [{**{"image": x}, **self.output_dims} 
                                    for x in batched_inputs]

        images = self.preprocess_image(batched_inputs)
        output = self.detection_transformer(images)
        ctrl_point_cls = output["pred_logits"]
        ctrl_point_coord = output["pred_ctrl_points"]
        ctrl_point_text = output["pred_text_logits"]
        bd_points = output["pred_bd_points"]

        return ctrl_point_cls, ctrl_point_coord, ctrl_point_text, bd_points

@META_ARCH_REGISTRY.register()
class ONNXExporterDetector(TransformerPureDetector):
    pass