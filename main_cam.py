import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import mediapipe as mp

# Load the trained model
model = load_model('data/my_model.h5')

# Class labels for sign language letters
class_names = ["A","B","C","D","E","F","G","H","I","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def predict_sign_language(frame):
    # Convert the frame to grayscale and resize to 28x28 as required by the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict the sign language letter
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    
    return class_names[predicted_class[0]]

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a natural view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Crop the region around the hand for prediction
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            
            h, w, _ = frame.shape
            x_min, y_min = int(x_min * w), int(y_min * h)
            x_max, y_max = int(x_max * w), int(y_max * h)
            
            # Ensure the coordinates are within the frame bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # Extract the hand region for prediction
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size != 0:
                predicted_letter = predict_sign_language(hand_img)
                # Display the predicted letter
                cv2.putText(frame, f"Prediction: {predicted_letter}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
