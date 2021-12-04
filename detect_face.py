import cv2
import mediapipe as mp
import numpy as np

def detect_faces(photo):
    '''
    :param photo: The real image
    :return: mapped values of normalizing_face and face_coords
    '''
    # face_coords is array of [x,y,w,h]
    face_coords = [where_is_face(photo)]
    print(face_coords)
    # cropped_face is the cropped ndarray of picture with colour intensity RGB
    if len(face_coords[0])>1:
        cropped_face = [photo[y:y + h, x:x + w] for x, y, w, h in face_coords]
        chk=1
        normalizing_face = [normalized_face(face,chk) for face in cropped_face]

        # returns the final ndarray of image
        if chk==0:
            return
        return zip(normalizing_face, face_coords)
    #else:
        #return zip([-1,-1,-1,-1],[-1,-1,-1,-1])
    #cropped_face=np.array(cropped_face)
    # normalizing_face is the cropped ndarray of picture with colour intensity gray
    


def normalized_face(face,chk):
    # convert the image to gray color
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # resize the image
        face = cv2.resize(face, (350, 350))
    except:
        chk=0
    # return the ndarray of image
    return face


def where_is_face(img):
    # detect the image and returns the co-ordinates of the face
    detector = FaceDetector()
    faces = detector.findFaces(img)
    return faces
    # faces will return (x,y,w,h) of the  face

class FaceDetector():
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
        try:
            return bboxs[0][1]
        except Exception as e:
            print(e)
            chk=0
            return tuple()