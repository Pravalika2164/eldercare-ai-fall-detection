from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np


class FallState(str, Enum):
    NORMAL = "NORMAL"
    POSSIBLE_FALL = "POSSIBLE FALL"
    CONFIRMED_FALL = "FALL DETECTED"
    RECOVERED = "RECOVERED"


@dataclass
class FallDecision:
    state: FallState
    score: float
    reason: str


class TemporalFallDetector:
    """
    Detects a physical fall candidate from temporal pose features.

    This detector does not confirm the final fall by itself.

    It looks for:
    1. rapid downward movement,
    2. horizontal / low body posture shortly afterwards,
    3. rejection of normal seated posture,
    4. recovery back to upright posture.

    Final confirmation is handled in app.py using
    ML probability and persistence.
    """

    def __init__(
        self,
        history_size: int = 45,
        drop_memory_seconds: float = 1.8,
        candidate_timeout: float = 3.0,
        recovery_seconds: float = 2.0,
    ):
        self.history = deque(maxlen=history_size)

        self.drop_memory_seconds = drop_memory_seconds
        self.candidate_timeout = candidate_timeout
        self.recovery_seconds = recovery_seconds

        self.state = FallState.NORMAL

        self.recent_drop_time = None
        self.possible_since = None
        self.recovery_since = None

    def update(
        self,
        features: np.ndarray,
        timestamp: float | None = None
    ) -> FallDecision:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        self.history.append(
            (now, features.copy())
        )

        if len(self.history) < 6:
            return FallDecision(
                self.state,
                0.0,
                "Collecting movement history"
            )

        current = features

        hip_y = float(current[3])
        torso_angle = float(current[10])
        aspect_ratio = float(current[11])
        bbox_height = float(current[13])

        previous_time, previous = self.history[-6]

        delta_time = max(
            now - previous_time,
            1e-3
        )

        hip_velocity = (
            hip_y - float(previous[3])
        ) / delta_time

        # ====================================================
        # COMPONENT SCORES
        # ====================================================

        horizontal_score = float(
            np.clip(
                0.55 * np.clip(
                    torso_angle / 0.75,
                    0.0,
                    1.0
                )
                +
                0.45 * np.clip(
                    aspect_ratio / 1.25,
                    0.0,
                    1.0
                ),
                0.0,
                1.0
            )
        )

        downward_score = float(
            np.clip(
                hip_velocity / 0.45,
                0.0,
                1.0
            )
        )

        low_position_score = float(
            np.clip(
                (hip_y - 0.48) / 0.30,
                0.0,
                1.0
            )
        )

        collapsed_height_score = float(
            np.clip(
                (0.65 - bbox_height) / 0.35,
                0.0,
                1.0
            )
        )

        rule_score = float(
            np.clip(
                0.35 * downward_score
                + 0.35 * horizontal_score
                + 0.20 * low_position_score
                + 0.10 * collapsed_height_score,
                0.0,
                1.0
            )
        )

        # ====================================================
        # TEMPORAL EVENTS
        # ====================================================

        rapid_drop = (
            downward_score > 0.45
        )

        # A real fall should usually end with
        # a strongly horizontal body configuration.
        horizontal_body = (
            torso_angle > 0.60
            or aspect_ratio > 1.05
        )

        fall_posture = (
            horizontal_body
            and horizontal_score > 0.68
            and low_position_score > 0.40
        )

        # Sitting lowers the hip position, but the torso
        # generally remains more vertical and the bounding box
        # remains relatively narrow.
        likely_sitting = (
            torso_angle < 0.50
            and aspect_ratio < 0.95
        )

        upright = (
            torso_angle < 0.38
            and aspect_ratio < 0.90
        )

        # ====================================================
        # REMEMBER RECENT RAPID DROP
        # ====================================================

        if rapid_drop:
            self.recent_drop_time = now

        recent_drop = (
            self.recent_drop_time is not None
            and (
                now - self.recent_drop_time
                <= self.drop_memory_seconds
            )
        )

        # ====================================================
        # STATE MACHINE
        # ====================================================

        if self.state == FallState.NORMAL:

            # Fall candidate requires:
            #
            # 1. recent fast downward movement
            # 2. horizontal/low body posture afterwards
            # 3. posture must NOT resemble normal sitting

            if (
                recent_drop
                and fall_posture
                and not likely_sitting
            ):
                self.state = FallState.POSSIBLE_FALL
                self.possible_since = now
                self.recovery_since = None

        elif self.state == FallState.POSSIBLE_FALL:

            # Final confirmation is handled by app.py.
            #
            # Reset if candidate lasts too long
            # without ML confirmation.

            if (
                self.possible_since is not None
                and (
                    now - self.possible_since
                    > self.candidate_timeout
                )
            ):
                self.reset_to_normal()

            # If person becomes upright again,
            # candidate was probably not a true fall.

            elif upright:
                self.reset_to_normal()

            # Additional seated-posture rejection.
            elif likely_sitting:
                self.reset_to_normal()

        elif self.state == FallState.CONFIRMED_FALL:

            if upright:

                if self.recovery_since is None:
                    self.recovery_since = now

                elif (
                    now - self.recovery_since
                    >= self.recovery_seconds
                ):
                    self.state = FallState.RECOVERED

            else:
                self.recovery_since = None

        elif self.state == FallState.RECOVERED:

            if upright:
                self.reset_to_normal()

        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        reason = (
            f"drop={downward_score:.2f}, "
            f"horizontal={horizontal_score:.2f}, "
            f"low={low_position_score:.2f}, "
            f"aspect={aspect_ratio:.2f}, "
            f"torso={torso_angle:.2f}, "
            f"sitting={int(likely_sitting)}, "
            f"recent_drop={int(recent_drop)}"
        )

        return FallDecision(
            self.state,
            rule_score,
            reason
        )

    def confirm_fall(self):
        """
        Called only by app.py after ML/persistence
        confirms the possible fall.
        """

        if self.state == FallState.POSSIBLE_FALL:
            self.state = FallState.CONFIRMED_FALL
            self.recovery_since = None

    def reset_to_normal(self):
        self.state = FallState.NORMAL

        self.possible_since = None
        self.recovery_since = None
        self.recent_drop_time = None