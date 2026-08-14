from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/processed/fall_sequences_annotated.npz")
MODEL_PATH = Path("models/fall_classifier_annotated.pkl")


def main():

    print("Loading processed dataset...")

    dataset = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = dataset["X"]
    y = dataset["y"]
    groups = dataset["groups"]

    print(f"Sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Fall samples: {np.sum(y == 1)}")
    print(f"Normal samples: {np.sum(y == 0)}")

    # Random Forest expects a 2D input.
    # Convert:
    #
    # (samples, 30 frames, 17 features)
    #
    # into:
    #
    # (samples, 510 features)

    X_flat = X.reshape(
        X.shape[0],
        -1
    )

    print(
        f"\nFlattened feature shape: {X_flat.shape}"
    )

    # IMPORTANT:
    # We split by original recording rather than individual
    # sequence windows.
    #
    # This prevents frames from the same recording appearing
    # in both training and testing data.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(
            X_flat,
            y,
            groups
        )
    )

    X_train = X_flat[train_index]
    X_test = X_flat[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    train_groups = groups[train_index]
    test_groups = groups[test_index]

    print("\nDATA SPLIT")
    print("-----------------------------")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Training recordings: "
        f"{len(np.unique(train_groups))}"
    )

    print(
        f"Testing recordings: "
        f"{len(np.unique(test_groups))}"
    )

    print(
        f"Training falls: {np.sum(y_train == 1)}"
    )

    print(
        f"Training normal: {np.sum(y_train == 0)}"
    )

    print(
        f"Testing falls: {np.sum(y_test == 1)}"
    )

    print(
        f"Testing normal: {np.sum(y_test == 0)}"
    )

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,

        # Important for our imbalanced dataset
        class_weight="balanced",

        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nMODEL RESULTS")
    print("=============================")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print("\nConfusion Matrix")

    print(matrix)

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Fall"
            ],
            digits=4,
            zero_division=0
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()