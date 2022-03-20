import cv2
from imread_from_url import imread_from_url

from hybridnets import HybridNets, optimized_model

model_path = "models/hybridnets_384x512/hybridnets_384x512.onnx"
anchor_path = "models/hybridnets_384x512/anchors_384x512.npy"

# Initialize road detector
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/2021-02-23_Tuesday_16.02.01-16.11.18_UTC-3_Route_S-40_%28Chile%29.webm/1920px--2021-02-23_Tuesday_16.02.01-16.11.18_UTC-3_Route_S-40_%28Chile%29.webm.jpg")

# Update road detector
seg_map, filtered_boxes, filtered_scores = roadEstimator(img)

combined_img = roadEstimator.draw_2D(img)
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Road Detections", combined_img)

cv2.imwrite("output.jpg", combined_img)
cv2.waitKey(0)