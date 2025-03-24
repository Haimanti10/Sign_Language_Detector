import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define 10 hand gestures
gesture_names = [
    "Shaswat", "Ayan", "Abhyuday", "Haimanti", "Hello", 
    "Yes", "No", "I", "Am", "Thankyou"
] # 5 fingers(open palm)=Shaswat, 2 fingers(victory)=Ayan, 3 fingers= Abhyuday, 4 fingers=Haimanti, Fist=Hello, Thumbsup=Yes, Thumbsdown=No, 
  #Index finger pointing up= I, Index finger pointing right= Am,, 5 fingers pointing right= Thankyou
number_of_classes= 10
dataset_size = 100  # Number of images per class

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

for class_id, gesture in enumerate(gesture_names):
    class_dir = os.path.join(DATA_DIR, str(class_id))  # Store by class index
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for "{gesture}". Press "Q" when ready.')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press "Q" to start capturing', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  # Start capturing images

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")
