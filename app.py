from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2

from src.alert_manager import AlertManager
from src.fall_detector import FallState, TemporalFallDetector
from src.feature_extractor import extract_features
from src.ml_classifier import MLSequenceClassifier
from src.pose_detector import PoseDetector


# ============================================================
# CONFIGURATION
# ============================================================

ML_THRESHOLD = 0.40
ML_CONFIRM_SECONDS = 0.60
CANCEL_SECONDS = 10.0

WINDOW_NAME = "ElderCare AI - Real-Time Fall Detection"


# ============================================================
# UTILITIES
# ============================================================

def parse_source(value: str):
    return int(value) if value.isdigit() else value


def percentage(value: float) -> str:
    value = max(0.0, min(1.0, value))
    return f"{value * 100:.0f}%"


# ============================================================
# UI
# ============================================================

def draw_panel(
    frame,
    state,
    combined_score,
    ml_probability,
    fps,
    reason,
    alert_countdown_active=False,
    remaining_seconds=0.0,
):
    """
    Draws the presentation-ready monitoring interface.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # MAIN HEADER
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 85),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "ELDERCARE AI",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Real-Time Elderly Fall Detection",
        (25, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (185, 185, 185),
        1
    )

    # Monitoring indicator
    cv2.circle(
        frame,
        (width - 155, 35),
        8,
        (0, 220, 0),
        -1
    )

    cv2.putText(
        frame,
        "MONITORING",
        (width - 135, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # STATUS COLORS
    # --------------------------------------------------------

    if state == FallState.CONFIRMED_FALL:
        state_color = (0, 0, 255)

    elif state == FallState.POSSIBLE_FALL:
        state_color = (0, 165, 255)

    elif state == FallState.RECOVERED:
        state_color = (255, 180, 0)

    else:
        state_color = (0, 210, 0)

    # --------------------------------------------------------
    # STATUS CARD
    # --------------------------------------------------------

    panel_top = height - 170

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, panel_top),
        (width, height),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0,
        frame
    )

    cv2.putText(
        frame,
        "SYSTEM STATUS",
        (25, panel_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (170, 170, 170),
        1
    )

    cv2.putText(
        frame,
        state.value,
        (25, panel_top + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        state_color,
        2
    )

    # --------------------------------------------------------
    # ML PROBABILITY
    # --------------------------------------------------------

    info_x = int(width * 0.42)

    cv2.putText(
        frame,
        "ML FALL PROBABILITY",
        (info_x, panel_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1
    )

    cv2.putText(
        frame,
        percentage(ml_probability),
        (info_x, panel_top + 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    # Probability bar
    bar_width = 150

    cv2.rectangle(
        frame,
        (info_x, panel_top + 80),
        (info_x + bar_width, panel_top + 94),
        (70, 70, 70),
        -1
    )

    probability_width = int(
        bar_width
        * max(0.0, min(ml_probability, 1.0))
    )

    if ml_probability >= ML_THRESHOLD:
        probability_color = (0, 140, 255)
    else:
        probability_color = (0, 200, 0)

    cv2.rectangle(
        frame,
        (info_x, panel_top + 80),
        (
            info_x + probability_width,
            panel_top + 94
        ),
        probability_color,
        -1
    )

    # --------------------------------------------------------
    # COMBINED SCORE / FPS
    # --------------------------------------------------------

    right_x = int(width * 0.72)

    cv2.putText(
        frame,
        "DETECTION SCORE",
        (right_x, panel_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1
    )

    cv2.putText(
        frame,
        percentage(combined_score),
        (right_x, panel_top + 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (right_x, panel_top + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1
    )

    # --------------------------------------------------------
    # TECHNICAL DEBUG LINE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        reason[:105],
        (25, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (160, 160, 160),
        1
    )

    # --------------------------------------------------------
    # ALERT COUNTDOWN OVERLAY
    # --------------------------------------------------------

    if alert_countdown_active:

        alert_overlay = frame.copy()

        cv2.rectangle(
            alert_overlay,
            (40, 105),
            (width - 40, 300),
            (20, 20, 20),
            -1
        )

        cv2.addWeighted(
            alert_overlay,
            0.92,
            frame,
            0.08,
            0,
            frame
        )

        cv2.rectangle(
            frame,
            (40, 105),
            (width - 40, 300),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "FALL DETECTED",
            (70, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "Are you OK?",
            (70, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Caregiver notification in "
            f"{remaining_seconds:.1f} sec",
            (70, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "PRESS C TO CANCEL",
            (70, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 220, 255),
            2
        )

    # --------------------------------------------------------
    # KEYBOARD HELP
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Q: Quit   |   C: Cancel caregiver alert",
        (width - 330, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 140, 140),
        1
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path"
    )

    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt"
    )

    parser.add_argument(
        "--no-alert",
        action="store_true"
    )

    parser.add_argument(
        "--use-ml",
        action="store_true"
    )

    parser.add_argument(
        "--ml-model",
        default="models/fall_classifier_annotated.pkl"
    )

    args = parser.parse_args()

    source = parse_source(
        args.source
    )

    # ========================================================
    # INITIALIZE COMPONENTS
    # ========================================================

    print("=" * 55)
    print(" ElderCare AI - Real-Time Fall Detection")
    print("=" * 55)

    print("[System] Loading YOLO pose detector...")

    pose_detector = PoseDetector(
        model_name=args.model
    )

    print("[System] Initializing temporal detector...")

    rule_detector = TemporalFallDetector()

    print("[System] Initializing alert manager...")

    alert_manager = AlertManager()

    ml_classifier = None

    if args.use_ml:

        print(
            "[System] Loading Random Forest classifier..."
        )

        ml_classifier = MLSequenceClassifier(
            args.ml_model
        )

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    print("[System] Opening camera...")

    capture = cv2.VideoCapture(
        source
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    print("[System] Monitoring started.")
    print("[System] Press Q to stop monitoring.")

    # ========================================================
    # SESSION STATE
    # ========================================================

    previous_time = time.time()

    alert_sent_for_current_fall = False

    ml_confirmation_start = None

    alert_countdown_active = False
    alert_countdown_start = None

    pending_image_path = None
    pending_timestamp_text = None
    pending_score = 0.0

    # ========================================================
    # LOGGING
    # ========================================================

    log_dir = Path(
        "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    session_time = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_path = log_dir / (
        f"live_predictions_{session_time}.csv"
    )

    log_file = open(
        log_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    log_writer = csv.writer(
        log_file
    )

    log_writer.writerow([
        "timestamp",
        "state",
        "combined_score",
        "ml_probability",
        "fps",
        "reason"
    ])

    print(
        f"[Logging] Predictions -> {log_path}"
    )

    # ========================================================
    # MAIN MONITORING LOOP
    # ========================================================

    try:

        while True:

            current_time = time.time()

            success, frame = capture.read()

            if not success:
                break

            # ------------------------------------------------
            # POSE DETECTION
            # ------------------------------------------------

            pose = pose_detector.detect(
                frame
            )

            display_frame = (
                frame.copy()
            )

            state = (
                rule_detector.state
            )

            score = 0.0
            rule_score = 0.0
            ml_probability = 0.0

            reason = (
                "No person detected"
            )

            if pose is not None:

                display_frame = (
                    pose.annotated_frame
                )

                # --------------------------------------------
                # FEATURE EXTRACTION
                # --------------------------------------------

                features = extract_features(
                    pose.keypoints,
                    pose.confidences,
                    pose.bbox,
                    frame.shape,
                )

                # --------------------------------------------
                # TEMPORAL RULE ENGINE
                # --------------------------------------------

                decision = (
                    rule_detector.update(
                        features
                    )
                )

                state = (
                    decision.state
                )

                rule_score = (
                    decision.score
                )

                score = (
                    rule_score
                )

                reason = (
                    decision.reason
                )

                # --------------------------------------------
                # RANDOM FOREST
                # --------------------------------------------

                if ml_classifier is not None:

                    ml_probability = (
                        ml_classifier.update(
                            features
                        )
                    )

                    score = (
                        0.55 * rule_score
                        + 0.45 * ml_probability
                    )

                    reason += (
                        f", ml="
                        f"{ml_probability:.2f}"
                    )

                    # ----------------------------------------
                    # ML FALL CONFIRMATION
                    # ----------------------------------------

                    if (
                        state
                        == FallState.POSSIBLE_FALL
                    ):

                        if (
                            ml_probability
                            >= ML_THRESHOLD
                        ):

                            if (
                                ml_confirmation_start
                                is None
                            ):

                                ml_confirmation_start = (
                                    current_time
                                )

                            elif (
                                current_time
                                - ml_confirmation_start
                                >= ML_CONFIRM_SECONDS
                            ):

                                rule_detector.confirm_fall()

                                state = (
                                    rule_detector.state
                                )

                                ml_confirmation_start = None

                        else:

                            ml_confirmation_start = None

                    else:

                        ml_confirmation_start = None

            # =================================================
            # FPS
            # =================================================

            fps = 1.0 / max(
                current_time
                - previous_time,
                1e-6
            )

            previous_time = (
                current_time
            )

            # =================================================
            # INCIDENT HANDLING
            # =================================================

            if (
                state
                == FallState.CONFIRMED_FALL
                and not alert_sent_for_current_fall
                and not alert_countdown_active
            ):

                timestamp = datetime.now()

                timestamp_file = (
                    timestamp.strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )
                )

                timestamp_text = (
                    timestamp.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                image_path = (
                    alert_manager.save_frame(
                        display_frame,
                        timestamp_file
                    )
                )

                print(
                    f"[Incident] Saved: "
                    f"{image_path}"
                )

                alert_sent_for_current_fall = True

                ml_confirmation_start = None

                if not args.no_alert:

                    # Local audible alarm
                    alert_manager.play_local_alarm(
                        duration_seconds=2
                    )

                    # Start cancellation countdown
                    alert_countdown_active = True

                    alert_countdown_start = (
                        time.time()
                    )

                    pending_image_path = (
                        image_path
                    )

                    pending_timestamp_text = (
                        timestamp_text
                    )

                    pending_score = (
                        score
                    )

                    print(
                        "[Alert] Fall detected."
                    )

                    print(
                        f"[Alert] Press C within "
                        f"{CANCEL_SECONDS:.0f} seconds "
                        f"if assistance is not required."
                    )

            # =================================================
            # CAREGIVER COUNTDOWN
            # =================================================

            remaining = 0.0

            if alert_countdown_active:

                elapsed = (
                    time.time()
                    - alert_countdown_start
                )

                remaining = max(
                    0.0,
                    CANCEL_SECONDS
                    - elapsed
                )

                if remaining <= 0:

                    print(
                        "[Alert] No cancellation received."
                    )

                    success = (
                        alert_manager.send_email(
                            subject=(
                                "URGENT: Possible "
                                "Fall Detected"
                            ),
                            message=(
                                "The elderly fall "
                                "detection system detected "
                                "a possible fall.\n\n"
                                f"Time: "
                                f"{pending_timestamp_text}\n"
                                f"Detection score: "
                                f"{pending_score:.0%}\n\n"
                                "The person did not cancel "
                                "the alert within "
                                "10 seconds.\n\n"
                                "Please check on them "
                                "as soon as possible."
                            ),
                            image_path=(
                                pending_image_path
                            )
                        )
                    )

                    if success:

                        print(
                            "[Alert] Caregiver notified."
                        )

                    alert_countdown_active = False
                    alert_countdown_start = None

            # =================================================
            # RESET AFTER RECOVERY
            # =================================================

            if (
                state in {
                    FallState.NORMAL,
                    FallState.RECOVERED
                }
                and not alert_countdown_active
            ):

                alert_sent_for_current_fall = False

            # =================================================
            # DRAW SHOWCASE UI
            # =================================================

            draw_panel(
                display_frame,
                state,
                score,
                ml_probability,
                fps,
                reason,
                alert_countdown_active,
                remaining
            )

            # =================================================
            # LOGGING
            # =================================================

            log_writer.writerow([
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                state.value,
                f"{score:.4f}",
                f"{ml_probability:.4f}",
                f"{fps:.2f}",
                reason
            ])

            log_file.flush()

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                display_frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # -------------------------------------------------
            # Q = QUIT
            # -------------------------------------------------

            if key == ord("q"):

                print(
                    "[System] Monitoring stopped by user."
                )

                break

            # -------------------------------------------------
            # C = CANCEL ALERT
            # -------------------------------------------------

            if (
                key == ord("c")
                and alert_countdown_active
            ):

                alert_countdown_active = False
                alert_countdown_start = None

                pending_image_path = None
                pending_timestamp_text = None
                pending_score = 0.0

                print(
                    "[Alert] Caregiver notification "
                    "cancelled by user."
                )

    finally:

        capture.release()

        cv2.destroyAllWindows()

        log_file.close()

        print(
            f"[Logging] Saved: "
            f"{log_path}"
        )

        print(
            "[System] ElderCare AI shutdown complete."
        )


if __name__ == "__main__":
    main()