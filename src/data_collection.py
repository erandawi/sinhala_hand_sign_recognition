import cv2
import mediapipe as mp
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

output_dir = 'data'
os.makedirs(output_dir, exist_ok = True)

landmarks_file = os.path.join(output_dir, 'landmarks.csv')
labels_file = os.path.join(output_dir, 'labels.csv')

current_label = None
current_label_count = 0
max_count_per_label = 1000
landmarks_data = []
labels_data = []

def save_to_csv(landmarks, label):
    global landmarks_data, labels_data
    
    landmarks_data.append(landmarks)
    labels_data.append([label])
    
    with open(landmarks_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(landmarks_data)
        
    with open(labels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(labels_data)
    
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened(): 
        success, image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

    # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            if current_label is not None and current_label_count < max_count_per_label:
                landmarks = [landmark for landmark in hand_landmarks.landmark]
                landmarks_flat = [item for sublist in [[lm.x, lm.y, lm.z] for lm in landmarks] for item in sublist]
                save_to_csv(landmarks_flat, current_label)
                current_label_count += 1
                
                if current_label_count >= max_count_per_label:
                    print(f"collected {max_count_per_label} point for label {current_label}")
                    current_label = None
                    current_label_count = 0
                
    # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    
        key = cv2.waitKey(5)
    
        if key == 27:
            break
        elif key != -1:
            current_label = chr(key)
            current_label_count = 0
            print(f"started collecting data for label {current_label}")
    
cap.release()
cv2.destroyAllWindows()