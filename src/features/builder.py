import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import pandas as pd

from src.core.stage import PipelineStage, RunContext
from src.core.timeline import Timeline
from src.core.feature_matrix import FeatureMatrix

logger = logging.getLogger("hypo_resilience.features.builder")

# Global registry of feature extraction functions
FEATURE_REGISTRY: List[Dict[str, Any]] = []


def register_feature(category: str, sources: List[str]):
    """
    Decorator to register a feature extractor function.
    
    The decorated function should accept a timeline dataframe and return a tuple:
    (features_dict, metadata_dict) where:
      - features_dict: Dict[feature_name, float_value]
      - metadata_dict: Dict[feature_name, Dict['description'/'units', str]]
    """
    def decorator(func: Callable[[pd.DataFrame], Tuple[Dict[str, float], Dict[str, Dict[str, str]]]]):
        FEATURE_REGISTRY.append({
            "func": func,
            "category": category,
            "sources": sources,
            "name": func.__name__
        })
        logger.debug(f"Registered feature extractor: {func.__name__} (Category: {category})")
        return func
    return decorator


class FeatureBuilder(PipelineStage):
    """
    Coordinates modular feature extraction from patient timelines using a registry.
    Combines all engineered features and saves them with a sidecar metadata JSON.
    """

    def __init__(self, timeline: Timeline, context: RunContext = None):
        super().__init__("feature_builder", context)
        self.timeline = timeline
        self.feature_matrix: FeatureMatrix | None = None

        # Resolve active categories from config or defaults
        if context and "features" in context.config:
            self.active_categories = context.config["features"].get(
                "active_categories",
                ["glucose", "insulin", "activity", "sleep", "nutrition", "variability", "circadian", "interaction"]
            )
        else:
            self.active_categories = [
                "glucose", "insulin", "activity", "sleep", "nutrition", "variability", "circadian", "interaction"
            ]

    def run(self) -> FeatureMatrix:
        """
        Runs all registered feature extractor functions on the patient timeline.
        """
        # Ensure all feature modules are imported to trigger registration
        self._import_feature_modules()

        features_combined = {}
        metadata_combined = {}

        df = self.timeline.df

        for entry in FEATURE_REGISTRY:
            category = entry["category"]
            if category not in self.active_categories:
                logger.debug(f"Skipping feature category '{category}' for extractor '{entry['name']}'")
                continue

            try:
                func = entry["func"]
                features, metadata = func(df)
                
                # Check for duplicate feature names
                for name, val in features.items():
                    if name in features_combined:
                        logger.warning(f"Feature name collision: '{name}' already exists. Overwriting.")
                    
                    features_combined[name] = val
                    
                    # Store metadata sidecar details
                    meta_details = metadata.get(name, {})
                    metadata_combined[name] = {
                        "category": category,
                        "sources": entry["sources"],
                        "description": meta_details.get("description", "No description provided"),
                        "units": meta_details.get("units", "None")
                    }
            except Exception as e:
                logger.exception(f"Error executing feature extractor '{entry['name']}': {e}")

        # Build feature matrix DataFrame (one row for this patient)
        features_df = pd.DataFrame([features_combined], index=[self.timeline.patient_id])
        features_df.index.name = "patient_id"

        self.feature_matrix = FeatureMatrix(features_df, metadata_combined)
        return self.feature_matrix

    def save(self) -> List[Path]:
        """Saves patient-level feature parquet and metadata json to the run directory."""
        if self.feature_matrix is None:
            logger.warning("FeatureMatrix has not been built yet. Save skipped.")
            return []

        features_dir = self.context.run_dir / "features" if self.context else Path("data/features")
        out_path = features_dir / f"{self.timeline.patient_id}.parquet"
        self.feature_matrix.save(out_path)
        return [out_path, out_path.with_suffix(".json")]

    def _import_feature_modules(self):
        """Imports all modular feature scripts to trigger their registration decorators."""
        import src.features.glucose
        import src.features.insulin
        import src.features.activity
        import src.features.sleep
        import src.features.nutrition
        import src.features.variability
        import src.features.circadian
        import src.features.interaction
