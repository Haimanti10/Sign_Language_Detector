import pickle
import pyttsx3
import cv2
import mediapipe as mp
import numpy as np
import threading

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()  # Run TTS asynchronously

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Label dictionary
labels_dict = {0: 'Shaswat', 1: 'Ayan', 2: 'Abhyuday', 3: 'Haimanti', 
    4: 'Hello', 5:'My', 6:'You', 7:'Is', 8: 'Name', 9:'Who', 10:'Are', 11:'Thankyou' }

# To prevent repeating audio for the same detection
last_prediction = ""

# To store detected words for scrolling
scrolling_text = ""

# Function to update scrolling text
def update_scrolling_text(new_word, scrolling_text, max_length=50):
    scrolling_text += f" {new_word}"
    if len(scrolling_text) > max_length:
        scrolling_text = scrolling_text[-max_length:]  # Keep only the latest part
    return scrolling_text

# Frame skipping variables
frame_count = 0
frame_skip = 3  # Only predict every 3 frames

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))  # Downscale frame for faster processing
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if frame_count % frame_skip == 0:  # Only process every nth frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Speak only if prediction changes
            if predicted_character != last_prediction:
                speak(predicted_character)
                last_prediction = predicted_character
                scrolling_text = update_scrolling_text(predicted_character, scrolling_text)

            # Display prediction on hand bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            last_prediction = ""  # Reset if no hand is detected

    # Display scrolling text on top of the screen with smaller font size
    cv2.rectangle(frame, (0, 0), (W, 20), (255, 255, 255), -1)  # Background for scrolling text
    cv2.putText(frame, scrolling_text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', cv2.resize(frame, (640, 480)))  # Upscale for display
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
