import argparse
from src.pipeline.orchestrator import main as orchestrator_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hypoglycemia Resilience Discovery pipeline using the orchestrator.")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to pipeline configuration YAML.")
    parser.add_argument("--skip", nargs="*", default=[], help="Stage names to skip.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.")
    args = parser.parse_args()
    orchestrator_main(["--config", args.config, "--skip"] + args.skip + (["--resume"] if args.resume else []))