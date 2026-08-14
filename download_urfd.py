from pathlib import Path
import time
import urllib.request
import zipfile


BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"

FALL_DIR = Path("data/raw/fall")
NORMAL_DIR = Path("data/raw/normal")

FALL_DIR.mkdir(parents=True, exist_ok=True)
NORMAL_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        with zipfile.ZipFile(path, "r") as zip_file:
            return zip_file.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_file(url: str, destination: Path, retries: int = 5) -> bool:
    if destination.exists():
        if is_valid_zip(destination):
            print(f"Already downloaded: {destination.name}")
            return True

        print(f"Removing corrupted file: {destination.name}")
        destination.unlink()

    for attempt in range(1, retries + 1):
        try:
            print(
                f"Downloading: {destination.name} "
                f"(attempt {attempt}/{retries})"
            )

            urllib.request.urlretrieve(url, destination)

            if is_valid_zip(destination):
                print(f"Saved successfully: {destination}")
                return True

            print("Downloaded file is corrupted.")

            if destination.exists():
                destination.unlink()

        except Exception as error:
            print(f"Download failed: {error}")

            if destination.exists():
                destination.unlink()

        if attempt < retries:
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)

    print(f"Could not download: {destination.name}")
    return False


def extract_zip(zip_path: Path, destination_folder: Path) -> None:
    extract_folder = destination_folder / zip_path.stem

    if extract_folder.exists() and any(extract_folder.iterdir()):
        print(f"Already extracted: {extract_folder.name}")
        return

    if not is_valid_zip(zip_path):
        print(f"Skipping invalid ZIP: {zip_path.name}")
        return

    print(f"Extracting: {zip_path.name}")

    extract_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_folder)

    print(f"Extracted: {extract_folder}")


def process_sequence(
    filename: str,
    destination_folder: Path
) -> None:

    url = f"{BASE_URL}/{filename}"
    destination = destination_folder / filename

    success = download_file(url, destination)

    if success:
        extract_zip(destination, destination_folder)


def download_falls() -> None:
    print("\n=== FALL SEQUENCES ===")

    for sequence_number in range(1, 31):

        filename = (
            f"fall-{sequence_number:02d}-cam0-rgb.zip"
        )

        process_sequence(
            filename,
            FALL_DIR
        )

        time.sleep(1)


def download_adls() -> None:
    print("\n=== ADL SEQUENCES ===")

    for sequence_number in range(1, 41):

        filename = (
            f"adl-{sequence_number:02d}-cam0-rgb.zip"
        )

        process_sequence(
            filename,
            NORMAL_DIR
        )

        time.sleep(1)


def main() -> None:

    print("\nUR Fall Detection Dataset Downloader")
    print("------------------------------------")

    download_falls()
    download_adls()

    print("\nFinished.")


if __name__ == "__main__":
    main()