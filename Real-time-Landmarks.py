import cv2
import numpy as np
import mediapipe as mp
import math
import os
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Create a directory to save the images
SAVE_DIR = "hand_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the Keras model
model = load_model('land01.h5')

# Define the labels
labels = ["0 Up", "1 Down", "2 Left", "3 Right", "4 Front", "5 Back"]


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


def process_image(img, imgSize, countdown, image_count):
    offset = 20
    predicted_label = None

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmark_x = [lm.x * img.shape[1] for lm in hand_landmarks.landmark]
            landmark_y = [lm.y * img.shape[0] for lm in hand_landmarks.landmark]
            min_x, min_y = min(landmark_x), min(landmark_y)
            max_x, max_y = max(landmark_x), max(landmark_y)

            # Calculate bounding box coordinates
            x, y, w, h = int(min_x) - offset, int(min_y) - offset, int(max_x - min_x) + 2 * offset, int(
                max_y - min_y) + 2 * offset

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Process imgWhite to detect hand landmarks
            white_image = draw_hand_landmarks(imgWhite, hands.process(
                cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)).multi_hand_landmarks, imgSize)

            # Display the live video and detected landmarks side by side
            cv2.imshow('Hand', imgWhite)
            cv2.imshow('Detected Landmarks', white_image)

            # Save white_image when countdown reaches "start"
            if countdown == 1:
                # Generate a unique filename based on timestamp
                filename = os.path.join(SAVE_DIR, f"hand_{image_count}.png")
                # Save the white_image
                cv2.imwrite(filename, white_image)
                print(f"Image {image_count} saved")

                # Load the saved image
                saved_image = cv2.imread(filename)
                # Preprocess the image for prediction
                preprocessed_image = cv2.resize(saved_image, (224, 224)) / 255.0
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                # Perform prediction using the model
                prediction = model.predict(preprocessed_image)[0]
                # Get the predicted label
                predicted_label = labels[np.argmax(prediction)]
                # Print the predicted label
                print(f"Predicted Label: {predicted_label}")



    # Countdown logic
    if countdown < 10:
        # Display the countdown number in green
        cv2.putText(img, str(countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Display "start" in red

        cv2.putText(img, "Start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img, predicted_label


def main():
    imgSize = 300
    cap = cv2.VideoCapture(0)

    countdown = 1
    image_count = 1  # Counter to keep track of saved images

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Failed to read from camera.")
            break

        img, pred = process_image(img, imgSize, countdown, image_count)

        cv2.imshow('Live Video', img)


        # Increment countdown or reset to 1 if it reaches 5
        countdown = countdown + 1 if countdown < 10 else 1



        # Check for "q" key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print(counter/total_count)
    # Release the VideoCapture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
