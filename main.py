import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / ( np.linalg.norm(ba) * np.linalg.norm(bc))

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def draw_skeleton(frame, landmarks):
    connections = [
        (11, 12),  # shoulders

        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm

        (11, 23), (12, 24),  # torso

        (23, 24),  # hips

        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    h, w = frame.shape[:2]

    for start, end in connections:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)

        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)

        cv2.line(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(frame,(x, y),4,(0, 255, 0),-1)


video_input_file = "video/sample.mp4"
video = cv2.VideoCapture(video_input_file)

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=10
)

landmarker = vision.PoseLandmarker.create_from_options(options)

while True:
    ok, frame = video.read()

    if not ok:
        break

    print("before detect")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    for person in result.pose_landmarks:

        draw_skeleton(frame, person)

        left_visibility = min(
            person[23].visibility,
            person[25].visibility,
            person[27].visibility
        )

        right_visibility = min(
            person[24].visibility,
            person[26].visibility,
            person[28].visibility
        )

        if left_visibility >= 0.5 and right_visibility >= 0.5:

            left_angle = calculate_angle(
                [person[23].x, person[23].y],
                [person[25].x, person[25].y],
                [person[27].x, person[27].y]
            )

            right_angle = calculate_angle(
                [person[24].x, person[24].y],
                [person[26].x, person[26].y],
                [person[28].x, person[28].y]
            )

            angle = (left_angle + right_angle) / 2

            if angle < 150:
                state = "Sitting"
            else:
                state = "Standing"

        elif left_visibility >= 0.5:
            angle = calculate_angle(
                [person[23].x, person[23].y],
                [person[25].x, person[25].y],
                [person[27].x, person[27].y]
            )

            state = "Sitting" if angle < 150 else "Standing"

        elif right_visibility >= 0.5:
            angle = calculate_angle(
                [person[24].x, person[24].y],
                [person[26].x, person[26].y],
                [person[28].x, person[28].y]
            )

            state = "Sitting" if angle < 150 else "Standing"

        else:
            state = "Unknown"

            
        nose = person[0]
        left_eye = person[2]
        right_eye = person[5]

        head_x = int(
            ((nose.x + left_eye.x + right_eye.x) / 3)
            * frame.shape[1]
        )

        head_y = int(
            ((nose.y + left_eye.y + right_eye.y) / 3)
            * frame.shape[0]
        )

        cv2.putText(
            frame,
            state,
            (head_x - 50, head_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

