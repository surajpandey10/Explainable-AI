import cv2
import numpy as np
import base64
from io import BytesIO
import json

# YOLO setup
net = cv2.dnn.readNet("weight/yolov3.weights", "cfg/yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_relationships = set()
    img_base64_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 != 0:
            continue

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        object_positions = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                object_positions.append((label, x, y, w, h))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Detect spatial relationships
        for i in range(len(object_positions)):
            obj1 = object_positions[i]
            for j in range(i + 1, len(object_positions)):
                obj2 = object_positions[j]
                obj1_label, obj1_x, obj1_y, obj1_w, obj1_h = obj1
                obj2_label, obj2_x, obj2_y, obj2_w, obj2_h = obj2
                if obj1_y < obj2_y:
                    all_relationships.add(f"{obj1_label} is above {obj2_label}")
                elif obj1_y > obj2_y:
                    all_relationships.add(f"{obj1_label} is below {obj2_label}")
                if obj1_x < obj2_x:
                    all_relationships.add(f"{obj1_label} is to the left of {obj2_label}")
                elif obj1_x > obj2_x:
                    all_relationships.add(f"{obj1_label} is to the right of {obj2_label}")

        # Convert the frame to base64
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_base64_list.append(img_base64)

    cap.release()
    return json.dumps({
        'relationships': list(all_relationships),
        'video_frames': img_base64_list
    })