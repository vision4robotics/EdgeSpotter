# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
# import multiprocessing as mp
import cv2
import tqdm

from detectron2.utils.logger import setup_logger

from predictor_video import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="EdgeSpotter Demo")
    parser.add_argument(
        "--config-file",
        default="configs/R_50/IPM/finetune_96voc_25maxlen.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="Path to video file.")
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window.",
    # )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        input_path = args.input[0]
        cam = cv2.VideoCapture(input_path)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis.get_image()[:, :, ::-1])
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

