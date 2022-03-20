import numpy as np
import cv2

segmentation_colors = np.array([[0,    0,    0],
								[255,  191,  0],
							 	[192,  67,   251]], dtype=np.uint8)

detection_color = (191,  255,  0)


def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
	color_segmap[seg_map>0] = segmentation_colors[seg_map[seg_map>0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img

def util_draw_detections(boxes, scores, image):

	for box, score in zip(boxes, scores):

		cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), detection_color, 1)

	return image

# Ref: https://github.com/datvuthanh/HybridNets/blob/d43b0aa8de2a1d3280084270d29cf4c7abf640ae/utils/utils.py#L615
def transform_boxes(boxes, anchors):

    y_centers_a = (anchors[:, 0] + anchors[:, 2]) / 2
    x_centers_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]

    w = np.exp(boxes[:, 3]) * wa
    h = np.exp(boxes[:, 2]) * ha

    y_centers = boxes[:, 0] * ha + y_centers_a
    x_centers = boxes[:, 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return np.vstack((xmin, ymin, xmax, ymax)).T


# Ref: https://python-ai-learn.com/2021/02/14/nmsfast/
def iou_np(box, boxes, area, areas):

    x_min = np.maximum(box[0], boxes[:,0])
    y_min = np.maximum(box[1], boxes[:,1])
    x_max = np.minimum(box[2], boxes[:,2])
    y_max = np.minimum(box[3], boxes[:,3])

    w = np.maximum(0, x_max - x_min + 1)
    h = np.maximum(0, y_max - y_min + 1)
    intersect = w*h
    
    iou_np = intersect / (area + areas - intersect)
    return iou_np

# Ref: https://python-ai-learn.com/2021/02/14/nmsfast/
def nms_fast(bboxes, scores, iou_threshold=0.5):
     
    areas = (bboxes[:,2] - bboxes[:,0] + 1) \
             * (bboxes[:,3] - bboxes[:,1] + 1)
    
    sort_index = np.argsort(scores)
    
    i = -1
    while(len(sort_index) >= 1 - i):

        max_scr_ind = sort_index[i]
        ind_list = sort_index[:i]

        iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
                     areas[max_scr_ind], areas[ind_list])
        print(max_scr_ind, ind_list, bboxes[max_scr_ind],bboxes[ind_list], iou)
        
        del_index = np.where(iou >= iou_threshold)
        sort_index = np.delete(sort_index, del_index)
        i -= 1
    
    bboxes = bboxes[sort_index]
    scores = scores[sort_index]
    
    return bboxes, scores