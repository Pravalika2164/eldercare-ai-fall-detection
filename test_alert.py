from src.alert_manager import AlertManager


def main():

    alert_manager = AlertManager(
        cooldown_seconds=0
    )

    print("Testing local alarm...")

    alert_manager.play_local_alarm(
        duration_seconds=2
    )

    print("Testing caregiver email...")

    success = alert_manager.send_email(
        subject=(
            "Fall Detection System - Test Alert"
        ),
        message=(
            "This is a test notification from "
            "the elderly fall detection system.\n\n"
            "The email alert system is working."
        )
    )

    if success:
        print(
            "Alert test completed successfully."
        )

    else:
        print(
            "Email test failed."
        )


if __name__ == "__main__":
    main()