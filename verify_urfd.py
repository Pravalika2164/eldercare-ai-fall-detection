from pathlib import Path


FALL_DIR = Path("data/raw/fall")
NORMAL_DIR = Path("data/raw/normal")


def count_sequence_folders(folder: Path, prefix: str):
    folders = [
        item for item in folder.iterdir()
        if item.is_dir() and item.name.startswith(prefix)
    ]
    return sorted(folders)


def count_pngs(folder: Path):
    return len(list(folder.rglob("*.png")))


def main():
    fall_folders = count_sequence_folders(FALL_DIR, "fall-")
    adl_folders = count_sequence_folders(NORMAL_DIR, "adl-")

    print("URFD DATASET CHECK")
    print("------------------")

    print(f"Fall sequence folders: {len(fall_folders)} / 30")
    print(f"ADL sequence folders:  {len(adl_folders)} / 40")

    print("\nPNG counts:")

    for folder in fall_folders:
        print(f"{folder.name}: {count_pngs(folder)} images")

    for folder in adl_folders:
        print(f"{folder.name}: {count_pngs(folder)} images")

    if len(fall_folders) == 30 and len(adl_folders) == 40:
        print("\nDataset structure looks complete.")
    else:
        print("\nSome sequences are missing.")


if __name__ == "__main__":
    main()