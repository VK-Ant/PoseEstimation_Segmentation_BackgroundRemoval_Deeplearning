import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from streamlit_extras.add_vertical_space import add_vertical_space

#sidebar contents

with st.sidebar:
    st.title('🤗VK-Pose Estimation and Object Detection using YOLOV8🤗')
    st.markdown('''
    ## About APP:
    
    - Person detection and pose estimation (Count the persons) 
    - YoloV8 Algorithm

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/venkat-vk/)
    
    ''')

    add_vertical_space(2)
    st.write('💡All about pose estimation, created by VK🤗')

model = YOLO('yolov8s-pose.pt')

cap = cv2.VideoCapture("forest.mp4")

st.title("VK-PERSON DETECTION & POSE ESTIMATION")

# Initialize a placeholder for the image to be displayed
image_placeholder = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, save=True)
        annotated = results[0].plot()
        
        # Convert the annotated frame to RGB format for Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Display the annotated frame in the Streamlit app
        image_placeholder.image(annotated_rgb, channels="RGB", width= 1000)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
