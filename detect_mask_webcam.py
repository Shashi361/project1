# USAGE
# python detect_mask_webcam.py

# import the necessary packages
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time
import streamlit as st
from imutils.video import VideoStream
# import argparse
import cv2


# import os

def mask_webcam():
    # facenet : find face model
    global color, label
    facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
    # model : model of detection mask
    model = load_model('models/mask_detector.model')

    # Reading webcam 
    cap = VideoStream(src=0).start()
    time.sleep(2.0)
    frameST = st.empty()
    # Reading webcam 
    # cap = cv2.VideoCapture(0)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(224, 224), mean=(104., 177., 123.))

        facenet.setInput(blob)
        dets = facenet.forward()

        # result_img = img.copy()
        frame = cap.read()
        frame = imutils.resize(frame, width=400)

        for i in range(dets.shape[2]):

            confidence = dets[0, 0, i, 2]

            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = frame[y1:y2, x1:x2]

            while True:
                try:
                    face_input = cv2.resize(face, dsize=(224, 224))
                    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                    face_input = preprocess_input(face_input)
                    face_input = np.expand_dims(face_input, axis=0)
                    break
                except:
                    print("resize error")
                    break

            (hmask, mask, nomask) = model.predict(face_input).squeeze()

            if mask > nomask and mask > hmask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)

            elif hmask > mask and hmask > nomask:
                color = (255, 0, 0)
                label = 'Haf Mask %d%%' % (hmask * 100)

            elif nomask > hmask:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            # cv2.putText(frame, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,color=color, thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow('Mask Detection',result_img)
        frameST.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.stop()


if __name__ == "__main__":
    mask_webcam()
