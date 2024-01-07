"""
For using this code as a library
Copy the codes under the main function and copy to new py file
Import the cv2, time and FaceDetectionModule(as fd)
Than add fd at the beginning of detector.
////
For changing the video or changing tracker to the webcam change the cap in main
"""

import cv2
import mediapipe as mp
import time


class FaceDetection():

    def __init__(self, min_detection_confidence=0.75, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_FaceDetection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_FaceDetection.FaceDetection(self.min_detection_confidence)

    def find_face(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mp_draw.draw_detection(img, detection) // Default Func
                bbboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbboxC.xmin * iw), int(bbboxC.ymin * ih), \
                    int(bbboxC.width * iw), int(bbboxC.height * ih)
                bboxs.append([bbox, detection.score])
                if draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f" {int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                (0, 255, 0))
        return bboxs, img

    def fancy_draw(self, img, bbox, l=30, t=3):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), 1)
        # Top Left x,y
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # Bottom Left x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        return img


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = FaceDetection()

    while True:
        success, img = cap.read()
        bboxs, img = detector.find_face(img)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"Fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
