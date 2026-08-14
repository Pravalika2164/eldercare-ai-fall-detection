import cv2

from src.feature_extractor import extract_features
from src.pose_detector import PoseDetector


def main():
    detector = PoseDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Feature test started. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        pose = detector.detect(frame)

        if pose is not None:
            output_frame = pose.annotated_frame

            features = extract_features(
                keypoints=pose.keypoints,
                confidences=pose.confidences,
                bbox=pose.bbox,
                frame_shape=frame.shape,
            )

            torso_angle_normalized = features[10]
            aspect_ratio = features[11]
            hip_y = features[3]
            body_height = features[13]

            torso_angle_degrees = torso_angle_normalized * 90

            cv2.putText(
                output_frame,
                f"Torso angle: {torso_angle_degrees:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output_frame,
                f"Hip Y: {hip_y:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output_frame,
                f"Aspect ratio: {aspect_ratio:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output_frame,
                f"Body height: {body_height:.2f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            print(
                f"torso={torso_angle_degrees:.1f} deg | "
                f"hip_y={hip_y:.2f} | "
                f"aspect_ratio={aspect_ratio:.2f} | "
                f"body_height={body_height:.2f}"
            )

        else:
            output_frame = frame

        cv2.imshow(
            "Fall Detection - Feature Test",
            output_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()