from pathlib import Path
import csv
import re

import cv2
import numpy as np

from src.feature_extractor import extract_features
from src.pose_detector import PoseDetector


FALL_DIR = Path("data/raw/fall")
NORMAL_DIR = Path("data/raw/normal")

FALL_ANNOTATION = FALL_DIR / "urfall-cam0-falls.csv"

OUTPUT_PATH = Path(
    "data/processed/fall_sequences_annotated.npz"
)

SEQUENCE_LENGTH = 30
STEP_SIZE = 5

# Require at least 80% of frames in a window
# to contain a successfully detected pose.
MIN_VALID_POSES = 24


def load_fall_annotations():
    """
    Reads URFD frame-level annotations.

    Dictionary structure:

    annotations["fall-01"][83] = 0
    annotations["fall-01"][120] = 1
    """

    annotations = {}

    with open(
        FALL_ANNOTATION,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) < 3:
                continue

            sequence_name = row[0].strip()

            frame_number = int(
                row[1].strip()
            )

            posture_label = int(
                row[2].strip()
            )

            if sequence_name not in annotations:
                annotations[sequence_name] = {}

            annotations[sequence_name][
                frame_number
            ] = posture_label

    return annotations


def extract_frame_number(image_path: Path):
    """
    Extract the frame number from the filename.

    Works with names containing a final numeric frame ID.
    """

    numbers = re.findall(
        r"\d+",
        image_path.stem
    )

    if not numbers:
        return None

    return int(numbers[-1])


def interpolate_missing_features(
    features
):
    """
    Replace occasional missing pose detections by
    linear interpolation across the sequence.
    """

    array = np.array(
        [
            feature
            if feature is not None
            else np.full(17, np.nan)
            for feature in features
        ],
        dtype=np.float32
    )

    for feature_index in range(
        array.shape[1]
    ):

        values = array[:, feature_index]

        missing = np.isnan(values)

        if not missing.any():
            continue

        valid = ~missing

        if valid.sum() < 2:
            return None

        values[missing] = np.interp(
            np.flatnonzero(missing),
            np.flatnonzero(valid),
            values[valid]
        )

        array[:, feature_index] = values

    return array


def determine_window_label(frame_labels):
    """
    Event-based fall labelling.

    URFD:
        -1 = normal/upright
         0 = falling transition
         1 = lying after fall

    Returns:
        1    -> fall event
        0    -> normal
        None -> ambiguous post-fall window; exclude from training
    """

    frame_labels = np.array(frame_labels)

    transition_count = np.sum(frame_labels == 0)
    lying_count = np.sum(frame_labels == 1)

    # Actual fall movement
    if transition_count >= 5:
        return 1

    # Fall transition followed by ground position
    if transition_count >= 2 and lying_count >= 3:
        return 1

    # Pure lying after a fall is NOT itself a fall event.
    # We exclude it so the model does not learn:
    # "lying down = fall".
    if transition_count == 0 and lying_count > 0:
        return None

    return 0



def process_fall_recording(
    folder,
    detector,
    annotation_map
):
    sequence_name = folder.name.split(
        "-cam0"
    )[0]

    image_paths = sorted(
        folder.rglob("*.png")
    )

    frame_features = []
    frame_labels = []

    sequence_annotations = (
        annotation_map.get(
            sequence_name,
            {}
        )
    )

    for image_path in image_paths:

        frame_number = extract_frame_number(
            image_path
        )

        if frame_number is None:
            continue

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        pose = detector.detect(
            frame
        )

        if pose is None:

            frame_features.append(
                None
            )

        else:

            feature_vector = extract_features(
                keypoints=pose.keypoints,
                confidences=pose.confidences,
                bbox=pose.bbox,
                frame_shape=frame.shape
            )

            frame_features.append(
                feature_vector
            )

        frame_label = sequence_annotations.get(
            frame_number,
            -1
        )

        frame_labels.append(
            frame_label
        )

    return (
        frame_features,
        frame_labels
    )


def process_normal_recording(
    folder,
    detector
):

    image_paths = sorted(
        folder.rglob("*.png")
    )

    frame_features = []
    frame_labels = []

    for image_path in image_paths:

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        pose = detector.detect(
            frame
        )

        if pose is None:

            frame_features.append(
                None
            )

        else:

            features = extract_features(
                keypoints=pose.keypoints,
                confidences=pose.confidences,
                bbox=pose.bbox,
                frame_shape=frame.shape
            )

            frame_features.append(
                features
            )

        # ADL is always normal
        frame_labels.append(-1)

    return (
        frame_features,
        frame_labels
    )


def create_windows(
    features,
    labels,
    group_name
):

    sequences = []
    sequence_labels = []
    sequence_groups = []

    maximum_start = (
        len(features)
        - SEQUENCE_LENGTH
        + 1
    )

    for start in range(
        0,
        maximum_start,
        STEP_SIZE
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        feature_window = (
            features[start:end]
        )

        label_window = (
            labels[start:end]
        )

        valid_pose_count = sum(
            feature is not None
            for feature in feature_window
        )

        if valid_pose_count < MIN_VALID_POSES:
            continue

        feature_window = (
            interpolate_missing_features(
                feature_window
            )
        )

        if feature_window is None:
            continue

        window_label = (
            determine_window_label(
                label_window
            )
        )

        if window_label is None:
            continue

        sequences.append(
            feature_window
        )

        sequence_labels.append(
            window_label
        )

        sequence_groups.append(
            group_name
        )

    return (
        sequences,
        sequence_labels,
        sequence_groups
    )


def get_sequence_folders(
    base_directory,
    prefix
):

    return sorted(
        folder
        for folder
        in base_directory.iterdir()

        if (
            folder.is_dir()
            and folder.name.startswith(
                prefix
            )
        )
    )


def main():

    print(
        "Loading URFD annotations..."
    )

    annotations = (
        load_fall_annotations()
    )

    print(
        f"Annotation recordings: "
        f"{len(annotations)}"
    )

    detector = PoseDetector()

    fall_folders = (
        get_sequence_folders(
            FALL_DIR,
            "fall-"
        )
    )

    normal_folders = (
        get_sequence_folders(
            NORMAL_DIR,
            "adl-"
        )
    )

    all_sequences = []
    all_labels = []
    all_groups = []

    print(
        "\nPROCESSING FALL RECORDINGS"
    )

    for folder in fall_folders:

        print(
            f"Processing {folder.name}"
        )

        features, labels = (
            process_fall_recording(
                folder,
                detector,
                annotations
            )
        )

        (
            sequences,
            sequence_labels,
            sequence_groups
        ) = create_windows(
            features,
            labels,
            folder.name
        )

        positive = sum(
            sequence_labels
        )

        negative = (
            len(sequence_labels)
            - positive
        )

        print(
            f"  Windows: {len(sequences)}"
        )

        print(
            f"  Fall: {positive}"
        )

        print(
            f"  Normal: {negative}"
        )

        all_sequences.extend(
            sequences
        )

        all_labels.extend(
            sequence_labels
        )

        all_groups.extend(
            sequence_groups
        )

    print(
        "\nPROCESSING ADL RECORDINGS"
    )

    for folder in normal_folders:

        print(
            f"Processing {folder.name}"
        )

        features, labels = (
            process_normal_recording(
                folder,
                detector
            )
        )

        (
            sequences,
            sequence_labels,
            sequence_groups
        ) = create_windows(
            features,
            labels,
            folder.name
        )

        print(
            f"  Windows: {len(sequences)}"
        )

        all_sequences.extend(
            sequences
        )

        all_labels.extend(
            sequence_labels
        )

        all_groups.extend(
            sequence_groups
        )

    X = np.array(
        all_sequences,
        dtype=np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int64
    )

    groups = np.array(
        all_groups
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        groups=groups
    )

    print(
        "\nANNOTATION-AWARE "
        "DATASET COMPLETE"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"Fall windows: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Normal windows: "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Total recordings: "
        f"{len(np.unique(groups))}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()