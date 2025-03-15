from edgespotter.onnx_model import SimpleONNXReadyModel
import torch
from post_process import post_prosses, TextVisualizer
import cv2
import torch.onnx
import cv2

CHECKPOINT = "../weights/ours.pth"
CONFIG = "configs/Base_det_export.yaml"
model = SimpleONNXReadyModel(CONFIG, CHECKPOINT)

cap = cv2.VideoCapture("../test_video/test1.mp4")
width = 960  
height = 540  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
DIMS = (960, 540)
while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_rgb = cv2.resize(frame, DIMS)

    original_image = frame_rgb[:, :, ::-1]
    frame_rgb = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

    predictions = model(frame_rgb)
    pr = post_prosses(predictions[0], predictions[1], predictions[2], predictions[3], DIMS)
    frame_rgb = frame_rgb.permute(1, 2, 0)
    visualizer = TextVisualizer(frame_rgb)
    vis_output = visualizer.draw_instance_predictions(predictions=pr)
    out_img = vis_output.get_image()[:, :, ::-1]
    out = cv2.resize(out_img, (960, 540))
    cv2.imshow('Camera', out)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()


