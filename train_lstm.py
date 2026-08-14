from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path(
    "data/processed/fall_sequences_annotated.npz"
)

MODEL_PATH = Path(
    "models/fall_lstm.pt"
)

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
PATIENCE = 6


class FallLSTM(nn.Module):
    def __init__(
        self,
        input_size=17,
        hidden_size=64,
        num_layers=2,
        dropout=0.30
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_timestep = output[:, -1, :]

        logits = self.classifier(
            last_timestep
        )

        return logits.squeeze(1)


def create_group_split(
    X,
    y,
    groups
):

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_val_index, test_index = next(
        first_split.split(
            X,
            y,
            groups
        )
    )

    X_train_val = X[train_val_index]
    y_train_val = y[train_val_index]
    groups_train_val = groups[train_val_index]

    X_test = X[test_index]
    y_test = y[test_index]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=24
    )

    train_index, val_index = next(
        second_split.split(
            X_train_val,
            y_train_val,
            groups_train_val
        )
    )

    X_train = X_train_val[train_index]
    y_train = y_train_val[train_index]

    X_val = X_train_val[val_index]
    y_val = y_train_val[val_index]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def standardize_data(
    X_train,
    X_val,
    X_test
):
    feature_mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    feature_std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    feature_std[
        feature_std < 1e-6
    ] = 1.0

    X_train = (
        X_train - feature_mean
    ) / feature_std

    X_val = (
        X_val - feature_mean
    ) / feature_std

    X_test = (
        X_test - feature_mean
    ) / feature_std

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32)
    )


def create_loader(
    X,
    y,
    shuffle
):
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


def evaluate_loss(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


def predict(
    model,
    loader,
    device
):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            batch_probabilities = (
                torch.sigmoid(logits)
            )

            probabilities.extend(
                batch_probabilities
                .cpu()
                .numpy()
            )

            labels.extend(
                y_batch.numpy()
            )

    probabilities = np.array(
        probabilities
    )

    labels = np.array(
        labels
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return (
        labels,
        predictions,
        probabilities
    )


def main():

    print(
        "Loading annotation-aware dataset..."
    )

    dataset = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = dataset["X"]
    y = dataset["y"]
    groups = dataset["groups"]

    print(
        f"Dataset shape: {X.shape}"
    )

    print(
        f"Fall windows: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Normal windows: "
        f"{np.sum(y == 0)}"
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = create_group_split(
        X,
        y,
        groups
    )

    print("\nDATA SPLIT")
    print("-------------------------")

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print(
        f"Training falls: "
        f"{np.sum(y_train == 1)}"
    )

    print(
        f"Training normal: "
        f"{np.sum(y_train == 0)}"
    )

    (
        X_train,
        X_val,
        X_test,
        mean,
        std
    ) = standardize_data(
        X_train,
        X_val,
        X_test
    )

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True
    )

    val_loader = create_loader(
        X_val,
        y_val,
        shuffle=False
    )

    test_loader = create_loader(
        X_test,
        y_test,
        shuffle=False
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            "cpu"
        )

    print(
        f"\nUsing device: {device}"
    )

    model = FallLSTM().to(
        device
    )

    fall_count = np.sum(
        y_train == 1
    )

    normal_count = np.sum(
        y_train == 0
    )

    positive_weight = (
        normal_count
        / max(fall_count, 1)
    )

    print(
        f"Positive class weight: "
        f"{positive_weight:.2f}"
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=device
            )
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None

    patience_counter = 0

    print("\nTRAINING")
    print("-------------------------")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_training_loss = 0
        total_samples = 0

        for (
            X_batch,
            y_batch
        ) in train_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = (
                X_batch.size(0)
            )

            total_training_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        training_loss = (
            total_training_loss
            / total_samples
        )

        validation_loss = (
            evaluate_loss(
                model,
                val_loader,
                criterion,
                device
            )
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"Train loss: "
            f"{training_loss:.4f} | "
            f"Val loss: "
            f"{validation_loss:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    model.load_state_dict(
        best_model_state
    )

    true_labels, predictions, probabilities = (
        predict(
            model,
            test_loader,
            device
        )
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    print("\nLSTM TEST RESULTS")
    print("=========================")

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

    print(
        confusion_matrix(
            true_labels,
            predictions
        )
    )

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            true_labels,
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

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "feature_mean":
                mean,

            "feature_std":
                std,

            "input_size":
                X.shape[2],

            "sequence_length":
                X.shape[1],
        },
        MODEL_PATH
    )

    print(
        f"\nBest LSTM saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()