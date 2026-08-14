from __future__ import annotations

import os
import smtplib
import time
import winsound
from email.message import EmailMessage
from pathlib import Path

import cv2
from dotenv import load_dotenv


class AlertManager:
    def __init__(
        self,
        incidents_dir: str = "incidents",
        cooldown_seconds: int = 60,
    ):
        load_dotenv()

        # Email configuration
        self.email_sender = os.getenv(
            "EMAIL_SENDER",
            ""
        ).strip()

        self.email_app_password = os.getenv(
            "EMAIL_APP_PASSWORD",
            ""
        ).strip()

        self.caregiver_email = os.getenv(
            "CAREGIVER_EMAIL",
            ""
        ).strip()

        # Incident storage
        self.incidents_dir = Path(
            incidents_dir
        )

        self.incidents_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Prevent repeated alerts
        self.cooldown_seconds = (
            cooldown_seconds
        )

        self.last_alert_time = 0.0

    def save_frame(
        self,
        frame,
        timestamp_text: str
    ) -> Path:

        filename = (
            f"fall_{timestamp_text}.jpg"
        )

        path = (
            self.incidents_dir
            / filename
        )

        cv2.imwrite(
            str(path),
            frame
        )

        return path

    def can_send(self) -> bool:
        return (
            time.time()
            - self.last_alert_time
            >= self.cooldown_seconds
        )

    def play_local_alarm(
        self,
        duration_seconds: int = 3
    ) -> None:
        """
        Plays a local audible alarm on Windows.
        """

        print(
            "[Alert] Local alarm activated."
        )

        end_time = (
            time.time()
            + duration_seconds
        )

        while time.time() < end_time:

            winsound.Beep(
                1500,
                400
            )

            time.sleep(
                0.15
            )

    def email_configured(self) -> bool:

        return all([
            self.email_sender,
            self.email_app_password,
            self.caregiver_email,
        ])

    def send_email(
        self,
        subject: str,
        message: str,
        image_path: Path | None = None
    ) -> bool:

        if not self.email_configured():

            print(
                "[Alert] Email credentials "
                "are not configured."
            )

            return False

        if not self.can_send():

            print(
                "[Alert] Email skipped "
                "because cooldown is active."
            )

            return False

        email = EmailMessage()

        email["From"] = (
            self.email_sender
        )

        email["To"] = (
            self.caregiver_email
        )

        email["Subject"] = subject

        email.set_content(
            message
        )

        # Attach incident image
        if (
            image_path is not None
            and image_path.exists()
        ):

            with image_path.open(
                "rb"
            ) as image_file:

                image_data = (
                    image_file.read()
                )

            email.add_attachment(
                image_data,
                maintype="image",
                subtype="jpeg",
                filename=image_path.name
            )

        try:

            with smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465
            ) as smtp:

                smtp.login(
                    self.email_sender,
                    self.email_app_password
                )

                smtp.send_message(
                    email
                )

            self.last_alert_time = (
                time.time()
            )

            print(
                "[Alert] Caregiver email sent."
            )

            return True

        except Exception as error:

            print(
                f"[Alert] Email failed: "
                f"{error}"
            )

            return False