import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
prev_time = 0

mp_FaceDetection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_FaceDetection.FaceDetection(min_detection_confidence=0.75)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection) // Default Func
            bbboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bbboxC.xmin * iw), int(bbboxC.ymin * ih), \
                int(bbboxC.width * iw), int(bbboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f" {int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0))

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f"Fps: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
