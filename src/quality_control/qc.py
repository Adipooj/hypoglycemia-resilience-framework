import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


def _checksum_file(file_path: Path) -> str:
    """Return SHA256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_quality_metrics(registry_path: Path, metadata_path: Path) -> dict:
    """Compute basic quality metrics from patient registry and dataset metadata.

    Returns a dictionary suitable for JSON/HTML reporting.
    """
    # Load registry
    df = pd.read_csv(registry_path)
    patient_count = df['patient_id'].nunique()
    # Load metadata
    with metadata_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    # Basic metrics
    metrics = {
        "patient_count": patient_count,
        "modalities": meta.get('modalities', []),
        "dataset_name": meta.get('dataset_name'),
        "dataset_version": meta.get('dataset_version'),
        "creation_timestamp": meta.get('creation_timestamp'),
        "checksum": meta.get('checksum'),
        "generated_timestamp": datetime.utcnow().isoformat() + 'Z',
    }
    return metrics


def write_quality_report(metrics: dict, output_dir: Path) -> None:
    """Write CSV and JSON quality reports, plus a simple HTML summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'quality_report.csv'
    json_path = output_dir / 'dataset_validation.json'
    html_path = output_dir / 'dataset_validation.html'

    # CSV (single row)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    # JSON
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    # HTML (basic table)
    html_content = f"""
    <html><head><title>Dataset Validation Report</title></head><body>
    <h1>Dataset Validation Report</h1>
    <table border='1'>
    {''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in metrics.items())}
    </table>
    </body></html>
    """
    html_path.write_text(html_content, encoding='utf-8')


def run_quality_control(project_root: Path) -> dict:
    """High‑level entry point used by the pipeline.

    Parameters
    ----------
    project_root: Path
        Root directory of the repository (contains `src/datasets`).
    """
    discovery_dir = project_root / 'src' / 'datasets'
    registry_path = discovery_dir / 'patient_registry.csv'
    metadata_path = discovery_dir / 'dataset_metadata.json'
    output_dir = project_root / 'reports' / 'qc'

    if not registry_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError('Discovery step must be run before quality control.')

    metrics = compute_quality_metrics(registry_path, metadata_path)
    write_quality_report(metrics, output_dir)
    logger.info(f"Quality control completed. Reports written to {output_dir}")
    return metrics

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    run_quality_control(root)
