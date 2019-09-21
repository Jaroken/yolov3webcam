# Script slighty modified from post: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
# Created video of first yolov3 test. Standard lite pretrained model used.
# Would like to use custom trained model next

import cv2
import numpy as numpy
import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('yolov3test1.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
font = cv2.FONT_HERSHEY_SIMPLEX
model = 'yolov3-tiny.weights'
config = 'yolov3-tiny.cfg'
yoloNet = cv2.dnn.readNetFromDarknet(config, model)
ln = yoloNet.getLayerNames()
ln = [ln[i[0] - 1] for i in yoloNet.getUnconnectedOutLayers()]

labelsPath = "C:\\Users\\PRESTK\\Documents\\darknet\\data\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

conf = 0.2
threshold = 0.2
(W, H) = (None, None)

while True:
    ret, frame = cap.read()
    if W is None or H is None: (H, W) =  frame.shape[:2]
    yoloNet.setInput(cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False))
    networkOutput = yoloNet.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in networkOutput:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    
    out.write(frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()