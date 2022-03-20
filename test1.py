import imutils
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import io
import time
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image
import gh_md_to_html
import ffmpeg
import imageio

# from moviepy.editor import*
# from moviepy.video.fx.resize import resize
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered',
                   initial_sidebar_state='expanded')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def mask_image():
    # load our serialized face detector model from disk
    global image
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["./face_detector/", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["./face_detector/",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return image, label


def mask_video():
    # global result_img
    cap = cv2.VideoCapture('./images/out_video.mp4')
    ret, img = cap.read()
    prototxtPath = os.path.sep.join(["./models/", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["./models/",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    facenet = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model('./models/mask_detector.model/')

    fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')  # need openh264
    out = cv2.VideoWriter('./images/out_video_result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (img.shape[1], img.shape[0]))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        result_img = img.copy()

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = img[y1:y2, x1:x2]

            face_input = cv2.resize(face, dsize=(224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            mask, nomask = model.predict(face_input).squeeze()

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)

            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=color, thickness=2, lineType=cv2.LINE_AA)

        out.write(result_img)
    out.release()
    cap.release()


def mask_webcam():
    # facenet : find face model
    global face_input
    prototxtPath = os.path.sep.join(["./models/", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["./models/",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    facenet = cv2.dnn.readNet(prototxtPath, weightsPath)
    # model : model of detection mask
    model = load_model('models/mask_detector.model')

    # app = Flask(__name__)
    # Reading webcam
    cap = VideoStream(src=0).start()
    time.sleep(2.0)
    frameST = st.empty()
    #cap = cv2.VideoCapture(0)
    #i = 0

    while True:
        frame = cap.read()


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

            try:
                face_input = cv2.resize(face, dsize=(224, 224))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0)
            except cv2.error as e:
                print("Invalid Frame!")
            cv2.waitKey()

            mask, nomask = model.predict(face_input).squeeze()

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
            #  winsound.Beep(frequency, duration)


            label = "{}: {:.2f}%".format(label, max(mask, nomask) * 100)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            #cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            #cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,color=color, thickness=2, lineType=cv2.LINE_AA)

        #cv2.imshow('Mask Detection', frame)
        frameST.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.stop()

def mask_detection():
    global vid_file
    #st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Video", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    choice = st.sidebar.selectbox("Select the type you want to detect", activities)

    if choice == 'Image':
        st.subheader("Detection on Image")
        image_file = st.file_uploader("Upload Image", type=['jpg'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='image uploaded successfully', use_column_width=True)
            if st.button('Process'):
                image, label = mask_image()
                st.image(image, width=420)  # display the uploaded image
                st.success('### ' + label)
                # st.success("success!")

    if choice == 'Video':
        st.subheader("Detection on Video")
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
        temporary_location = False

        if uploaded_file is not None:
            our_video = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            st.video(our_video, start_time=0)
            temporary_location = "./images/out_video.mp4"

            with open(temporary_location, 'wb') as out1:  ## Open temporary file as bytes
                out1.write(our_video.read())  ## Read bytes into file

            # close file
            # out.close()
            if st.button('Process'):
                mask_video()
                vid_file = open("./images/out_video_result.mp4", "rb").read()
                st.video(vid_file, start_time=0)

    if choice == 'Webcam':
        st.subheader("Detection on Webcam")
        if st.button('Process'):
            mask_webcam()
            # st.success("success")


mask_detection()
