import time
import cv2
import numpy as np


image_path = 'C9.png'                 ######path to image
config_path = 'yolov3.cfg'              #####path to config file (.cfg)
weight_path = 'yolov3.weights'          #####path to weights file
class_name_path = 'yolov3.txt'          #####path_to_class_name

####note: Nếu dùng yolov4.cfg thì phải dùng yolov4.weights và ngược lại, class name thì giữ nguyên vì 2 phiên bản đều có class name giống nhau

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = (0,255,0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(img, str(round(confidence,2)), (x , y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


image = cv2.imread(image_path)

Width = image.shape[1]
Height = image.shape[0]
scale = 1 / 255

classes = None

with open(class_name_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


net = cv2.dnn.readNet(weight_path, config_path)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.2
nms_threshold = 0.4

# Thực hiện xác định bằng HOG và SVM
start = time.time()

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("object detection", image)


end = time.time()
print("YOLO Execution time: " + str(end-start))


cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
