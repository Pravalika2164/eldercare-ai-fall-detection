from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np


class MLSequenceClassifier:
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 30
    ):
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Run train_model.py first."
            )

        self.model = joblib.load(self.model_path)

        # Use a single worker during real-time inference.
        # The Random Forest was trained with n_jobs=-1,
        # which can cause repeated joblib/sklearn warnings
        # during continuous prediction.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

    def update(self, features: np.ndarray) -> float:
        self.buffer.append(
            features.copy()
        )

        if len(self.buffer) < self.sequence_length:
            return 0.0

        sequence = np.array(
            self.buffer,
            dtype=np.float32
        ).reshape(1, -1)

        if hasattr(
            self.model,
            "predict_proba"
        ):
            return float(
                self.model.predict_proba(
                    sequence
                )[0, 1]
            )

        return float(
            self.model.predict(
                sequence
            )[0]
        )