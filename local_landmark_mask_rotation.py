"""
Add
1.  from face_alignment import *
2.  mask rotation part

"""

import sys
import cv2
import numpy as np
import dlib
import argparse
from face_alignment import *
# Hyperparameter
resize_size = 200.

# global variable
texture_id = 0

# Initiate dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D model points.
modelPoints = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
size = [resize_size, resize_size]
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camMat = np.array(
[[focal_length, 0, center[0]],
[0, focal_length, center[1]],
[0, 0, 1]], dtype = "double"
)


dist_coef = np.zeros((4,1)) # Assuming no lens distortion


DEBUGMODE = 1
MASKMODE = 2

SURGICAL_MASK = 1

# debug parameters
MODE = MASKMODE

surgical_mask = cv2.imread('./res/mask/surgical_mask.png', cv2.IMREAD_UNCHANGED)
rows, cols, channels = surgical_mask.shape


# function from https://stackoverflow.com/a/54058766
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def resize_image(inputImg, width=None, height=None):
    org_width, org_height, _ = inputImg.shape
    if width is None and height is None:
        print("Error as both width and height is NONE")
        exit(-1)
    elif width is None:
        ratio = 1.0 * height / org_height
        width = int(ratio * org_width)
        dim = (width, height)
    else:
        ratio = 1.0 * width / org_width
        height = int(ratio * org_height)
        dim = (width, height)
    resized = cv2.resize(inputImg, dim, interpolation=cv2.INTER_AREA)
    return resized



class Cam:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.curFrame = self.capture.read()[1]

    def update_frame(self):
        self.curFrame = self.capture.read()[1]
    
    def get_curFrame(self):
        return self.curFrame


class Detector:
    def __init__(self):
        self.feature = [] # saves face rectangles and landmarks
        self.org_feature = [] # feature elements transformed to fit original window
        self.org_nosePoint = [] # coordinates for noses
        self.transMat = []
        self.rotMat = []

    def resetInternals(self):
        self.feature = []
        self.org_feature = []
        self.org_nosePoint = []
        self.transMat = []
        self.rotMat = [] 

    # Detect landmark and headpose
    def detect(self, frame):
        self.resetInternals()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = resize_size / gray.shape[1]
        dim = (int(resize_size), int(gray.shape[0] * ratio))
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        rects = detector(resized, 1)
        for i, rect in enumerate(rects):
            t = rect.top()
            b = rect.bottom()
            l = rect.left()
            r = rect.right()
            org_t = int(t/ratio)
            org_b = int(b/ratio)
            org_l = int(l/ratio)
            org_r = int(r/ratio)

            self.feature.append({})
            self.feature[i]['rect'] = [t, b, l ,r]
            self.feature[i]['landmark'] = []

            self.org_feature.append({})
            self.org_feature[i]['rect'] = [org_t, org_b, org_l ,org_r]
            self.org_feature[i]['landmark'] = []
            
            # Detect landmark
            shape = predictor(resized, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                org_x, org_y = int(x/ratio), int(y/ratio)
                self.feature[i]['landmark'].append([x, y])
                self.org_feature[i]['landmark'].append([org_x, org_y])
                if MODE == DEBUGMODE:
                    cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
            
            # Detect headpose
            nose_tip = self.feature[i]['landmark'][33]
            chin = self.feature[i]['landmark'][8]
            # left_eye_left_corner = self.feature[i]['landmark'][45]
            # right_eye_right_corner = self.feature[i]['landmark'][36]
            left_mouth_corner = self.feature[i]['landmark'][54]
            right_mouth_corner = self.feature[i]['landmark'][48]
            right_eye_right_corner = self.feature[i]['landmark'][45]
            left_eye_left_corner = self.feature[i]['landmark'][36]
            right_mouth_corner = self.feature[i]['landmark'][54]
            left_mouth_corner = self.feature[i]['landmark'][48]
            image_points = np.array([
                (nose_tip[0], nose_tip[1]),
                (chin[0], chin[1]),
                (left_eye_left_corner[0], left_eye_left_corner[1]),
                (right_eye_right_corner[0], right_eye_right_corner[1]),
                (left_mouth_corner[0], left_mouth_corner[1]),
                (right_mouth_corner[0], right_mouth_corner[1])
            ], dtype="double")

            _, self.rotMat, self.transMat = cv2.solvePnP(modelPoints, image_points, camMat, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
            nosePoints, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), self.rotMat, self.transMat, camMat, dist_coef)

            # print("Transmat:")
            # print(self.transMat)
            # print("rotmat:")
            # print(self.rotMat)
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nosePoints[0][0][0]), int(nosePoints[0][0][1]))
            org_p1 = (int(p1[0] / ratio), int(p1[1] / ratio))
            org_p2 = (int(p2[0] / ratio), int(p2[1] / ratio))
            self.org_nosePoint.append([])
            self.org_nosePoint[-1].append(org_p1)
            self.org_nosePoint[-1].append(org_p2)

            if MODE == DEBUGMODE:
                cv2.line(resized, p1, p2, (255,0,0), 2)


            if MODE == DEBUGMODE and len(self.feature) > 0:
                cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
                resized = cv2.flip(resized, 1)
                cv2.imshow('resized', resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def get_feature(self):
        return self.feature
    
    def get_org_feature(self):
        return self.org_feature
    
    def get_org_nosePoint(self):
        return self.org_nosePoint


class FaceMask:
    def __init__(self):
        self.cam = Cam()
        self.detector = Detector()

    def update_frame(self):
        self.cam.update_frame()
        self.detector.detect(self.cam.get_curFrame())
    
    def show_frame(self, maskType=None, showMask=False):
        global surgical_mask
        curFrame = self.cam.get_curFrame()
        if MODE == DEBUGMODE:
            landmarks = self.detector.get_org_feature()
            # print(landmarks)
            for i in range(len(landmarks)):
                for j in range(68):
                    cv2.circle(curFrame, (landmarks[i]['landmark'][j][0], landmarks[i]['landmark'][j][1]), 1, (0, 0, 255), -1)
                t = landmarks[i]['rect'][0]
                b = landmarks[i]['rect'][1]
                l = landmarks[i]['rect'][2]
                r = landmarks[i]['rect'][3]
                cv2.rectangle(curFrame, (l, t), (r, b), (0, 255, 0), 2)
                point = self.detector.get_org_nosePoint()
                cv2.line(curFrame, point[i][0], point[i][1], (255,0,0), 2)
            curFrame = cv2.flip(curFrame, 1)
            cv2.imshow('original', curFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        elif MODE == MASKMODE and showMask == True:
            landmarks = self.detector.get_org_feature()
            # print(landmarks)
            for i in range(len(landmarks)):
                landmark = landmarks[i]['landmark']
                if maskType == SURGICAL_MASK:
                    magic_num = 10
                    left_cheek_x = landmark[2][0] - magic_num
                    right_cheek_x = landmark[14][0] + magic_num
                    upper_lip_y = landmark[27][1]
                    width = right_cheek_x - left_cheek_x
                    mask = surgical_mask.copy()
 
                    #####################################   mask rotation start #####################################
                    h, w = mask.shape[:2]
                    angle=get_angle(landmark[48], landmark[54])
                    mask = rotate_opencv(mask, (w/2, h/2), -angle)
                    mask = resize_image(mask, width=width)
                    #####################################   mask rotation end #######################################

                    curFrame = overlay_transparent(curFrame, mask, left_cheek_x + magic_num, upper_lip_y)
            curFrame = cv2.flip(curFrame, 1)
            ret, jpeg = cv2.imencode('.jpg', curFrame)
            return jpeg.tobytes()
            # curFrame = cv2.flip(curFrame, 1)
            # cv2.imshow('original', curFrame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     exit()
        elif showMask == False:
            curFrame = cv2.flip(curFrame, 1)
            ret, jpeg = cv2.imencode('.jpg', curFrame)
            return jpeg.tobytes()

    def main(self):
        global surgical_mask
        while True:
            self.update_frame()
            self.show_frame(maskType=SURGICAL_MASK)

 
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=False, type=int, default=2, help="Debug mode: 1, Mask mode: 2")
    args = vars(ap.parse_args())
    if args['mode'] == DEBUGMODE:
        print("Debug mode")
    elif args['mode'] == MASKMODE:
        print("Mask mode")
    MODE = args['mode']
    faceMask = FaceMask()
    faceMask.main()
