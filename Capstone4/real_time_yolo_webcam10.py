import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("weight/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to get spatial relationships
def get_spatial_relationship(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    relationship = ""

    if y1 < y2:
        relationship += "above "
    elif y1 > y2:
        relationship += "below "
    if x1 < x2:
        relationship += "to the left of"
    elif x1 > x2:
        relationship += "to the right of"

    return relationship.strip()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Store detected objects for relationship detection
    detected_objects = []

    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_text = f"{int(confidences[i] * 100)}%"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence_text}", (x, y - 10), font, 2, color, 2)

            detected_objects.append((label, (x, y, w, h)))
    else:
        cv2.putText(frame, "No objects detected", (20, 30), font, 2, (0, 0, 255), 2)

    # Calculate and display spatial relationships
    for i in range(len(detected_objects)):
        for j in range(i + 1, len(detected_objects)):
            obj1_label, obj1_box = detected_objects[i]
            obj2_label, obj2_box = detected_objects[j]
            relationship = get_spatial_relationship(obj1_box, obj2_box)
            if relationship:  # Only show if a relationship is found
                print(f"{obj1_label} is {relationship} {obj2_label}")

    # Display the resulting frame
    cv2.imshow("Real-time Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
