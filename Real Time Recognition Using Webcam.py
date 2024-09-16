#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import model_from_json
import cv2
import numpy as np
from datetime import datetime

# Load the model
json_file = open("signlanguagemodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagemodel.h5")

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 64, 64, 1)  # Use 64x64 as per the trained model
    return feature / 255.0

# Function to check if current time is within the allowed range
def is_within_time_range(start_hour, end_hour):
    now = datetime.now()
    return start_hour <= now.hour < end_hour

# Time range for operation (6 PM to 10 PM)
START_HOUR = 18
END_HOUR = 22

if is_within_time_range(START_HOUR, END_HOUR):
    cap = cv2.VideoCapture(0)
    label = ['A', 'B', 'C', 'del', 'nothing', 'space']

    while True:
        _, frame = cap.read()
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        cropframe = frame[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (64, 64))  # Resize to 64x64
        cropframe = extract_features(cropframe)
        pred = model.predict(cropframe)
        
        prediction_label = label[pred.argmax()]
        
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            accu = "{:.2f}".format(np.max(pred) * 100)
            cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("output", frame)
        if cv2.waitKey(27) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("This system operates only between 6 PM and 10 PM.")


# In[ ]:




