import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# Directory constants (relative to repository root)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "datasets"

def _hash_file(file_path: Path) -> str:
    """Return SHA256 hash of a file's contents (binary)."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def discover_modality_files() -> Dict[str, List[Path]]:
    """Scan ``RAW_DATA_DIR`` for modality files.

    Returns a dictionary mapping modality names (derived from the immediate
    sub‑directory name) to a list of absolute :class:`~pathlib.Path` objects.
    Files are expected to be in Parquet or CSV format; other extensions are
    ignored.
    """
    modalities: Dict[str, List[Path]] = {}
    if not RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
    for modality_dir in RAW_DATA_DIR.iterdir():
        if not modality_dir.is_dir():
            continue
        modality = modality_dir.name
        files = [p for p in modality_dir.rglob("*.*")
                 if p.suffix.lower() in {".parquet", ".csv"}]
        modalities[modality] = files
    return modalities

def generate_patient_registry(modality_files: Dict[str, List[Path]]) -> Path:
    """Create ``patient_registry.csv`` that lists unique patient identifiers.

    The function extracts patient IDs from file names assuming a convention
    ``<patient_id>_*.parquet`` or ``<patient_id>_*.csv``.  The resulting CSV has
    a single column ``patient_id`` sorted alphabetically.
    """
    patient_ids: set[str] = set()
    for files in modality_files.values():
        for f in files:
            stem = f.stem
            # Assume patient ID is the first token before an underscore
            pid = stem.split("_")[0]
            patient_ids.add(pid)
    registry_path = OUTPUT_DIR / "patient_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="") as csvfile:
        csvfile.write("patient_id\n")
        for pid in sorted(patient_ids):
            csvfile.write(f"{pid}\n")
    return registry_path

def create_dataset_metadata(
    dataset_name: str,
    dataset_version: str,
    loader_version: str,
    modality_files: Dict[str, List[Path]],
    patient_registry_path: Path,
) -> Path:
    """Generate ``dataset_metadata.json`` containing provenance information.

    The JSON includes:

    * ``dataset_name`` – human readable name
    * ``dataset_version`` – version string supplied by the user
    * ``loader_version`` – version of the loader code (derived from ``__version__`` if available)
    * ``modalities`` – list of modality names
    * ``patient_count`` – number of unique patients
    * ``creation_timestamp`` – ISO‑8601 UTC timestamp
    * ``checksum`` – SHA‑256 of the concatenated hashes of all raw files
    * ``git_commit`` – placeholder (filled at runtime by the pipeline)
    """
    # Compute a combined checksum of all raw files for reproducibility
    combined_hash = hashlib.sha256()
    for files in modality_files.values():
        for f in sorted(files):
            combined_hash.update(_hash_file(f).encode())
    metadata = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "loader_version": loader_version,
        "modalities": list(modality_files.keys()),
        "patient_count": len(open(patient_registry_path).read().splitlines()) - 1,
        "creation_timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "checksum": combined_hash.hexdigest(),
        "git_commit": "${GIT_COMMIT_HASH}",  # to be substituted by the pipeline runner
    }
    meta_path = OUTPUT_DIR / "dataset_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return meta_path

def main() -> None:
    """Entry point used by the pipeline script.

    The function performs the three steps in order and prints the locations of
    the generated artifacts.  All paths are absolute to make downstream modules
    independent of the working directory.
    """
    modalities = discover_modality_files()
    registry_path = generate_patient_registry(modalities)
    meta_path = create_dataset_metadata(
        dataset_name="Hypoglycemia Resilience Cohort",
        dataset_version="1.0.0",
        loader_version="1.0.0",
        modality_files=modalities,
        patient_registry_path=registry_path,
    )
    print(f"Patient registry written to: {registry_path}")
    print(f"Dataset metadata written to: {meta_path}")

if __name__ == "__main__":
    main()
