import mediapipe as mp
import numpy as np
import cv2
import glob

def cap_pictures():
    cap = cv2.VideoCapture(0)
    detector = Face_Detection()
    img_counter=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break

        k = cv2.waitKey(1)
        frame, bboxs = detector.findFaces(frame)
        
        if k%256 == 27:
            # ESC pressed
            print("Escape button pressed. Exiting")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imshow("Image_Cap" , frame)
            cv2.waitKey(0)
            img_name = "Image_{}.png".format(img_counter)
            emotion=emotions[img_counter//3]
            cv2.imwrite("data/new/%s/%s.png" % (emotion, img_counter), frame)  # write image
            img_counter += 1
            #cv2.imwrite(img_name, frame)
            #print("{} written!".format(img_name))
            

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
    cap_pictures()
    