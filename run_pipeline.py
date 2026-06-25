from src.ingestion.dataset_scanner import scan_dataset

manifest = scan_dataset()

print()

print("=" * 60)

print("Patients discovered:", len(manifest))

print("=" * 60)

for pid in sorted(manifest):

    print(f"{pid:<6} {list(manifest[pid].keys())}")