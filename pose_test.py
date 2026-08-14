import cv2
from ultralytics import YOLO


def start_pose_detection() -> None:
    model = YOLO("yolo11n-pose.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "The webcam could not be opened. Check your camera permission."
        )

    print("Webcam started. Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the webcam.")
            break

        results = model.predict(
            source=frame,
            conf=0.4,
            verbose=False,
        )

        output_frame = results[0].plot()

        cv2.putText(
            output_frame,
            "Human Pose Detection",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Elderly Fall Detection Project",
            output_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_pose_detection()