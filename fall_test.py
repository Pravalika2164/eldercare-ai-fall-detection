import cv2

from src.fall_detector import TemporalFallDetector
from src.feature_extractor import extract_features
from src.pose_detector import PoseDetector


def main():
    pose_detector = PoseDetector()
    fall_detector = TemporalFallDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Fall detection test started.")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        pose = pose_detector.detect(frame)

        if pose is not None:
            output_frame = pose.annotated_frame

            features = extract_features(
                keypoints=pose.keypoints,
                confidences=pose.confidences,
                bbox=pose.bbox,
                frame_shape=frame.shape,
            )

            decision = fall_detector.update(features)

            state = decision.state.value
            score = decision.score

            if state == "FALL DETECTED":
                status_color = (0, 0, 255)

            elif state == "POSSIBLE FALL":
                status_color = (0, 165, 255)

            elif state == "RECOVERED":
                status_color = (255, 200, 0)

            else:
                status_color = (0, 255, 0)

            cv2.rectangle(
                output_frame,
                (10, 10),
                (620, 120),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                output_frame,
                f"Status: {state}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
            )

            cv2.putText(
                output_frame,
                f"Fall score: {score:.2f}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output_frame,
                decision.reason,
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )

        else:
            output_frame = frame

            cv2.putText(
                output_frame,
                "No person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow(
            "Real-Time Fall Detection Test",
            output_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()