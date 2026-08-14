import cv2

from src.pose_detector import PoseDetector


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


IMPORTANT_POINTS = [
    5,   # left shoulder
    6,   # right shoulder
    11,  # left hip
    12,  # right hip
    13,  # left knee
    14,  # right knee
    15,  # left ankle
    16,  # right ankle
]


def main():
    detector = PoseDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Keypoint test started. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        pose = detector.detect(frame)

        if pose is not None:
            output_frame = pose.annotated_frame

            for index in IMPORTANT_POINTS:
                x, y = pose.keypoints[index]
                confidence = pose.confidences[index]

                if confidence > 0.4:
                    print(
                        f"{KEYPOINT_NAMES[index]:15} "
                        f"x={x:.1f} "
                        f"y={y:.1f} "
                        f"confidence={confidence:.2f}"
                    )

        else:
            output_frame = frame

        cv2.imshow(
            "Fall Detection - Keypoint Test",
            output_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()