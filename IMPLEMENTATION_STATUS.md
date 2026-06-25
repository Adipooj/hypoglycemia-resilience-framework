# Project Progress

Overall completion percentage: 8%
Current phase: Research & Planning
Current module: Architecture & Config Design
Estimated remaining work: 92%

---

# Completed Tasks

* [x] Inspected current workspace and existing files.
* [x] Checked structure and schemas of UOM raw data modalities.
* [x] Drafted initial architecture transition plan.
* [x] Updated design plan with user feedback (Canonical adapter, plugins, registries, experiment tracking, feature metadata, and YAML configurations).

---

# Files Created

None

---

# Files Modified

None

---

# Public APIs

Standard APIs to be implemented and locked:

```python
# In datasets/uom_loader.py
loader = UOMLoader()
patient = loader.load("2301")

# In timeline/builder.py
timeline = TimelineBuilder(patient)
timeline.run()
timeline.save()

# In features/builder.py
features = FeatureBuilder(timeline)
features.run()
features.save()

# In phenotype/clustering.py
clusters = PhenotypeDiscovery(features)
clusters.run()
clusters.save()

# In models/trainer.py or models/predictor.py
model = ResilienceModel(features)
model.run()
model.save()

# In explainability/shap_engine.py
explainer = Explainability(model)
explainer.run()
explainer.save()
```

---

# Architectural Decisions

1. **Restructuring to Locked Architecture**: 
   - Move `src/ingestion/patient.py` to `src/core/patient.py`.
   - Move/refactor `src/ingestion/patient_loader.py` and `dataset_scanner.py` to `src/datasets/base_loader.py` and `src/datasets/uom_loader.py`.
   - Refactor `src/preprocessing/quality_control.py` to `src/preprocessing/quality_controller.py`.
   - Refactor `src/preprocessing/timeline.py` to `src/timeline/builder.py` and implement modular timeline components for glucose, insulin, activity, nutrition, and sleep.
   - Refactor `src/preprocessing/datetime_utils.py` to `src/preprocessing/datetime_utils.py` (kept in preprocessing).
   - This ensures strict adherence to the Locked Architecture.

2. **Glucose Unit Normalization**:
   - The UOM dataset glucose values are in mmol/L. Since clinical indices (MAGE, LBGI, HBGI, ADRR) are standardly defined in mg/dL, we will convert glucose internally to mg/dL for feature computation where needed, or support both.

3. **Ingestion Layer Retention**:
   - Keep the ingestion folder `src/ingestion` and its scripts to orchestrate CSV directory scans and data loader integration.

4. **Canonical Schema Adapter**:
   - Implement a schema adapter between the loaders and `TimelineBuilder` to standardize columns and types, decoupling loaders from timeline structure.

5. **Plugin-Based Timeline Mergers**:
   - Each modality merger (glucose, insulin, sleep, activity, nutrition) registers itself to the `TimelineBuilder` as a plugin, allowing modular alignment logic.

6. **Registry-Based Feature Builder**:
   - Features will be registered using a decorator registry (`@register_feature`), allowing individual features to define their own metadata (description, units, sources) and run parameters.

7. **Experiment Tracking & Immutable Runs**:
   - Runs will be recorded inside a dedicated `runs/run_YYYYMMDD_HHMMSS` directory, containing the active YAML configuration, output parquets, logs, and reporting outputs.

8. **YAML-Driven Orchestration**:
   - The pipeline execution order and hyperparameters will be configured dynamically via `configs/pipeline_config.yaml`.

---

# Remaining Tasks

## Core
- [ ] Implement `src/core/stage.py` (Base class for stages)
- [ ] Implement `src/core/patient.py`
- [ ] Implement `src/core/timeline.py`
- [ ] Implement `src/core/feature_matrix.py`

## Ingestion & Adapters
- [ ] Implement Canonical Adapter layer
- [ ] Refactor `src/ingestion/patient_loader.py` to support canonical validation

## Datasets
- [ ] Implement `src/datasets/base_loader.py`
- [ ] Implement `src/datasets/uom_loader.py`
- [ ] Implement dummy/stub/skeleton loaders for `diatrend_loader.py` and `openaps_loader.py` to support future datasets.

## Preprocessing
- [ ] Implement `src/preprocessing/quality_controller.py`
- [ ] Implement `src/preprocessing/validators.py`
- [ ] Implement `src/preprocessing/datetime_utils.py`

## Timeline Builder (Plugin-based)
- [ ] Implement `src/timeline/builder.py`
- [ ] Implement modular timeline merges:
  - [ ] `src/timeline/glucose.py`
  - [ ] `src/timeline/insulin.py`
  - [ ] `src/timeline/activity.py`
  - [ ] `src/timeline/nutrition.py`
  - [ ] `src/timeline/sleep.py`

## Feature Engineering (Registry-based)
- [ ] Implement `src/features/builder.py`
- [ ] Implement modular feature extraction (~100 features total) and metadata generation:
  - [ ] `src/features/glucose.py`
  - [ ] `src/features/insulin.py`
  - [ ] `src/features/activity.py`
  - [ ] `src/features/sleep.py`
  - [ ] `src/features/nutrition.py`
  - [ ] `src/features/variability.py`
  - [ ] `src/features/circadian.py`
  - [ ] `src/features/interaction.py`

## Phenotype Discovery
- [ ] Implement `src/phenotype/clustering.py`
- [ ] Implement `src/phenotype/evaluation.py`
- [ ] Implement `src/phenotype/resilience_score.py`

## Modeling
- [ ] Implement `src/models/trainer.py` (XGBoost, LightGBM, CatBoost, Random Forest)
- [ ] Implement `src/models/predictor.py`

## Explainability
- [ ] Implement `src/explainability/shap_engine.py`

---

# Current State

* What was the LAST thing completed? Revised implementation plan to incorporate user's mandatory feedback.
* What file was being edited? IMPLEMENTATION_STATUS.md.
* What is the VERY NEXT thing that should be implemented? Creation of pipeline config and base modules in `src/core/` and `src/datasets/`.
* If implementation stopped right now due to token limits, exactly where should work resume? Awaiting user approval of the revised implementation plan.
