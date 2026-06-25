"""
Physiological and technical constants for the Hypoglycemia Resilience project.
"""

# ==============================================================================
# Glucose Thresholds & Conversion Constants
# ==============================================================================

# Unit conversion factor from mmol/L to mg/dL
MMOL_TO_MGDL = 18.0182

# Glucose thresholds in mmol/L
GLUCOSE_HYPO_LEVEL2 = 3.0       # < 54 mg/dL
GLUCOSE_HYPO_LEVEL1 = 3.9       # < 70 mg/dL
GLUCOSE_TARGET_LOW = 3.9        # 70 mg/dL
GLUCOSE_TARGET_HIGH = 10.0      # 180 mg/dL
GLUCOSE_HYPER_LEVEL1 = 10.0     # > 180 mg/dL
GLUCOSE_HYPER_LEVEL2 = 13.9     # > 250 mg/dL

# Glucose thresholds in mg/dL (clinical formulas)
GLUCOSE_HYPO_MGDL = 70.0
GLUCOSE_HYPER_MGDL = 180.0

# ==============================================================================
# Activity Intensity Thresholds (METs)
# ==============================================================================
MET_SEDENTARY_MAX = 1.5
MET_LIGHT_MAX = 3.0
MET_MODERATE_MAX = 6.0
# Anything above MET_MODERATE_MAX is vigorous

# ==============================================================================
# Preprocessing & Quality Control Thresholds
# ==============================================================================
MIN_GLUCOSE_DAYS = 5
MIN_GLUCOSE_DATA_PCT = 0.70      # Require at least 70% glucose readings for validity
EXPECTED_READINGS_PER_DAY = 288  # 24 hours * 12 (5-min intervals)
