#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json
from datetime import datetime

# Load the model
json_file = open("signlanguagedetectionmodel48x483.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x483.h5")

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

if is_within_time_range(START_HOUR, END_HOUR):
    # GUI using Tkinter
    class SignLanguageApp:
        def __init__(self, master):
            self.master = master
            master.title("Sign Language Detection")
            master.geometry("400x500")

            self.label = tk.Label(master, text="Upload an Image for Prediction", font=("Arial", 16))
            self.label.pack(pady=20)

            self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
            self.upload_button.pack(pady=10)

            self.result_label = tk.Label(master, text="", font=("Arial", 16))
            self.result_label.pack(pady=10)

            self.image_label = tk.Label(master)
            self.image_label.pack(pady=10)

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

    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
else:
    # Display a message if outside the allowed time range
    print("This application operates only between 6 PM and 10 PM.")


# In[ ]:




