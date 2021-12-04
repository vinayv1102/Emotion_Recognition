"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL
import cv2
from detect_face import detect_faces
from image_enhance import nparray_as_image, draw_with_alpha
import mediapipe as mp
import numpy as np
import time

'''
def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param emoticons: List of emotions images.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    while read_value:
        for normalized_face, (x, y, w, h) in detect_faces(webcam_image):
            prediction = model.predict(normalized_face)  # do prediction
            if cv2.__version__ != '3.1.0':
                prediction = prediction[0]

            image_to_draw = emoticons[prediction]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)
        '''

def show_webcam_and_run(model):
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = Face_Detection()
    img_counter=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break
        #cv2.imshow("Video", frame)
 
        k = cv2.waitKey(1)
        frame, bboxs = detector.findFaces(frame)
        
        if k%256 == 27:
            # ESC pressed
            print("Escape button pressed. Exiting")
            break
        elif k%256 == 32:
            # SPACE pressed
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for normalized_face, (x, y, w, h) in detect_faces(frame):
                prediction = model.predict(normalized_face)  # do prediction
            '''
            vertices=bboxs[0][1]
            x=vertices[0]
            y=vertices[1]
            w=vertices[2]
            h=vertices[3]
            crop_img = img[y:y+h, x:x+w]
            cv2.imshow("cropped", crop_img)
            crop_img = cv2.resize(crop_img, (350, 350))
            '''
            cv2.putText(normalized_face, f'Prediction: {prediction[0]}', (70, 70), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 2)
            cv2.imshow("Prediction" , normalized_face)
            print("Prediction:",emotions[prediction[0]])
            cv2.waitKey(0)
            img_name = "Prediction_{}.png".format(img_counter)
            cv2.imwrite(img_name, normalized_face)
            print("{} written!".format(img_name))
            img_counter += 1
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Video", frame)

    cap.release()

    cv2.destroyAllWindows()
    
class Face_Detection():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                '''if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)'''
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

if __name__ == '__main__':
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    #emotions = ['anger', 'disgust', 'happy', 'sadness', 'surprise']
    # load model
    fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read('models/emotion_detection_model.xml')

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(fisher_face)