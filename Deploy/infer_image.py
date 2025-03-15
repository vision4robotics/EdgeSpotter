from edgespotter.onnx_model import SimpleONNXReadyModel
import time 
import torch
from post_process import post_prosses, TextVisualizer
import cv2
import torch.onnx
import cv2

CHECKPOINT = "model_0012000.pth"
CONFIG = "configs/Base_det_export.yaml"

model = SimpleONNXReadyModel(CONFIG, CHECKPOINT)

DIMS = (960, 540)
frame = cv2.imread("image_0098.jpg")

frame_rgb = cv2.resize(frame, DIMS)

original_image = frame_rgb[:, :, ::-1]
frame_rgb = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

t = time.time()
predictions = model(frame_rgb)
pr = post_prosses(predictions[0], predictions[1], predictions[2], predictions[3], DIMS)
frame_rgb = frame_rgb.permute(1, 2, 0)
visualizer = TextVisualizer(frame_rgb)
vis_output = visualizer.draw_instance_predictions(predictions=pr)
out_img = vis_output.get_image()[:, :, ::-1]
out = cv2.resize(out_img, (960, 540))
cv2.imshow('Camera', out)
cv2.waitKey (0) 
cv2.destroyAllWindows()
