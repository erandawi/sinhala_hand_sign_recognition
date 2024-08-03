from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


model = tf.keras.models.load_model('model/hand_sign_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('data/classes.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
frame = None
prediction = None

sinhala_letters = {
    1: 'අ',
    2: 'ආ',
    3: 'ඉ'
}

def preprocess_landmarks(landmarks):
    landmarks = np.array(landmarks).flatten()
    landmarks = landmarks.reshape(1, 21, 3, 1)
    return landmarks

def generate_frames():
    global frame, prediction
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    landmarks = preprocess_landmarks(landmarks)
                    prediction = model.predict(landmarks)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    global prediction
    if prediction is not None:
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        predicted_number = int(predicted_label)
        predicted_letter = sinhala_letters.get(predicted_number, 'Unknown') 
        return jsonify({'prediction': predicted_letter})
    return jsonify({'prediction': 'No hand detected'})

if __name__ == '__main__':
    app.run(debug=True)
