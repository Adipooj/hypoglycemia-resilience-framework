from pathlib import Path
import json
import re

from src.config import UOM_ROOT
from src.config import INTERIM_DATA


FOLDERS = {
    "activity": UOM_ROOT / "Activity Data",
    "glucose": UOM_ROOT / "Glucose Data",
    "nutrition": UOM_ROOT / "Nutrition Data",
    "demographics": UOM_ROOT / "Demographics",
    "sleep": UOM_ROOT / "Sleep Data",
    "basal": UOM_ROOT / "Insulin Data" / "Basal Data",
    "bolus": UOM_ROOT / "Insulin Data" / "Bolus Data",
}


def extract_patient_id(filename: str):

    match = re.search(r"(\d{4})", filename)

    return match.group(1) if match else None


def scan_dataset():

    manifest = {}

    for modality, folder in FOLDERS.items():

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        for file in folder.glob("*.csv"):

            pid = extract_patient_id(file.name)

            if pid is None:
                continue

            manifest.setdefault(pid, {})

            manifest[pid][modality] = str(file)

    outfile = INTERIM_DATA / "manifest.json"

    with open(outfile, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n{len(manifest)} patients discovered.")

    print(f"Manifest written to:\n{outfile}")

    return manifest


if __name__ == "__main__":

    scan_dataset()