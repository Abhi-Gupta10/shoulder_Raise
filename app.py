import os
import cv2
import mediapipe as mp
import numpy as np
import threading
from flask import Flask, Response, render_template
from playsound import playsound
from gtts import gTTS

app = Flask(__name__)

# Generate and save audio feedback
tts_up = gTTS("Up", lang="en")
tts_down = gTTS("Down", lang="en")
tts_up.save("upsh_new.mp3")
tts_down.save("downsh_new.mp3")

# Function to calculate angles using 2D coordinates
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle

# Function to play sound asynchronously
def play_sound(sound_file):
    threading.Thread(target=lambda: playsound(sound_file), daemon=True).start()

# Initialize MediaPipe Pose
mp_drawing, mp_pose = mp.solutions.drawing_utils, mp.solutions.pose

# Open camera
cap = cv2.VideoCapture(0)
counter, stage = 0, None

def generate_frames():
    global counter, stage
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    if right_angle > 150 and left_angle > 150 and stage != "up":
                        stage = "up"
                        play_sound("upsh_new.mp3")

                    if right_angle < 30 and left_angle < 30 and stage == "up":
                        stage = "down"
                        counter += 1
                        play_sound("downsh_new.mp3")

                    cv2.putText(image, f'Count: {counter}', (480, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Stage: {stage}', (480, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

            _, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
