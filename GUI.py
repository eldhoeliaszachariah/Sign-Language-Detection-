import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json
from datetime import datetime
import threading

# Load the model
json_file = open("signlanguagemodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagemodel.h5")

# Labels for predictions
label = ['A', 'B', 'C', 'del', 'nothing', 'space']

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 64, 64, 1)  # Use 64x64 as per the trained model
    return feature / 255.0

# Function to predict sign language
def predict_sign_language(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))  # Resize to 64x64
    features = extract_features(image)
    pred = model.predict(features)
    prediction_label = label[pred.argmax()]
    accuracy = np.max(pred) * 100
    return prediction_label, accuracy

# Function to check if current time is within the allowed range
def is_within_time_range(start_hour, end_hour):
    now = datetime.now()
    return start_hour <= now.hour < end_hour

# Time range for operation (6 PM to 10 PM)
START_HOUR = 18
END_HOUR = 22

class SignLanguageApp:
    def __init__(self, master):
        self.master = master
        master.title("Sign Language Detection")
        master.geometry("600x600")

        self.label = tk.Label(master, text="Upload an Image or Use Real-time Video", font=("Arial", 16))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.video_button = tk.Button(master, text="Start Video Detection", command=self.start_video_thread)
        self.video_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop Video Detection", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        self.cap = None
        self.video_running = False

    def upload_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            # Load and display image in the GUI
            img = Image.open(image_path)
            img = img.resize((200, 200))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

            # Make prediction and display the result
            prediction_label, accuracy = predict_sign_language(image_path)
            self.result_label.config(text=f'Prediction: {prediction_label}\nAccuracy: {accuracy:.2f}%')

    def start_video_thread(self):
        if not self.video_running and is_within_time_range(START_HOUR, END_HOUR):
            self.video_running = True
            self.video_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            threading.Thread(target=self.start_video_detection).start()

    def start_video_detection(self):
        self.cap = cv2.VideoCapture(0)
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            crop_frame = frame[40:300, 0:300]
            crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            crop_frame = cv2.resize(crop_frame, (64, 64))  # Resize to 64x64
            features = extract_features(crop_frame)
            pred = model.predict(features)

            prediction_label = label[pred.argmax()]
            accuracy = np.max(pred) * 100

            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            cv2.putText(frame, f'{prediction_label}  {accuracy:.2f}%', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Real-time Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
        self.video_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if is_within_time_range(START_HOUR, END_HOUR):
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
else:
    print("This application operates only between 6 PM and 10 PM.")