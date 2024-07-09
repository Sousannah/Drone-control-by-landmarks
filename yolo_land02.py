import cv2
import numpy as np
import mediapipe as mp
import math
import time
from tensorflow.keras.models import load_model
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Load the YOLOv4 model
net = cv2.dnn.readNet("cross-hands-yolov4-tiny.weights", "cross-hands-yolov4-tiny.cfg")

# Define class labels
class_labels = ["0 Up", "1 Down", "2 Left", "3 Right", "4 Front", "5 Back"]


def detect_hands(frame):
    height, width, _ = frame.shape

    # Prepare the input blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get detection results
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Lists to store detected hands
    hand_boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3 and class_id == 0:  # Adjust the threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the top-left coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                hand_boxes.append((x, y, w, h))

    return hand_boxes

def draw_hand_landmarks(image, hand_landmarks, imgSize):
    white_image = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            for landmark in hand_landmark.landmark:
                x, y = int(landmark.x * imgSize), int(landmark.y * imgSize)
                cv2.circle(white_image, (x, y), 5, (0, 255, 0), -1)

            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                x0, y0 = int(hand_landmark.landmark[connection[0]].x * imgSize), int(
                    hand_landmark.landmark[connection[0]].y * imgSize)
                x1, y1 = int(hand_landmark.landmark[connection[1]].x * imgSize), int(
                    hand_landmark.landmark[connection[1]].y * imgSize)
                cv2.line(white_image, (x0, y0), (x1, y1), (0, 0, 0), 3)

    return white_image

def classify_hand_gesture(white_image, model, labels):
    # Resize image to match the expected input shape of the model
    resized_image = cv2.resize(white_image, (224, 224))

    # Classify the image using the loaded model
    prediction = model.predict(np.expand_dims(resized_image, axis=0))

    if prediction.size > 0:
        # Map the prediction to the class labels
        label_index = np.argmax(prediction)
        if label_index < len(labels):
            predicted_label = labels[label_index]
            return predicted_label
        else:
            print("Error: Invalid label index.")
            return None
    else:
        print("Error: Empty prediction.")
        return None



# Create directory if it doesn't exist
if not os.path.exists('hands'):
    os.makedirs('hands')

def process_image(frame, imgSize, model, labels):
    hand_boxes = detect_hands(frame)

    for hand_box in hand_boxes:
        x, y, w, h = hand_box

        # Expand the bounding box to make it bigger
        x -= 20  # Subtract 20 pixels from the x-coordinate
        y -= 20  # Subtract 20 pixels from the y-coordinate
        w += 40  # Add 40 pixels to the width
        h += 40  # Add 40 pixels to the height

        # Draw bounding box around the detected hand on the main screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        hand = frame[y:y+h, x:x+w]

        # Convert the BGR image to RGB
        hand_rgb = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(hand_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                white_image = draw_hand_landmarks(hand, [hand_landmarks], imgSize)

                # If hand landmarks are detected, classify the hand gesture
                if white_image is not None:
                    # Save the image
                    img_name = f"hands/{int(time.time())}.png"
                    cv2.imwrite(img_name, white_image)

                    # Predict hand gesture
                    predicted_label = classify_hand_gesture(white_image, model, labels)
                    print("Predicted Label:", predicted_label)

                    # Display hand with landmarks in the second screen
                    cv2.imshow('Hand with Landmarks', white_image)

    # Display the frame with detected hand on the main screen
    cv2.imshow('Detected Hand', frame)

    return frame

def main():
    imgSize = 300
    cap = cv2.VideoCapture(0)

    # Load the Keras model
    model = load_model('land01.h5')

    # Create windows for displaying hand with landmarks and detected hand
    cv2.namedWindow('Hand with Landmarks', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detected Hand', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Failed to read from camera.")
            break

        frame = process_image(frame, imgSize, model, class_labels)

        cv2.imshow('Live Video', frame)

        # Check for key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the VideoCapture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
