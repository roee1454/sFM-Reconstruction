"""
Global settings and configuration for the reconstruction pipeline.
"""

import os

# ============================================================================
# QUALITY PROFILE CONFIGURATION
# ============================================================================
# Change this to switch between reconstruction profiles
# Options: "SPEED", "BALANCED", or "QUALITY"
QUALITY_PROFILE = "SPEED"
# ============================================================================

# Path to OpenMVS binaries (relative to project root)
OPENMVS_BIN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "thirdparty",
    "OpenMVS",
    "bin"
)

# Default paths
DEFAULT_DATASET_PATH = "test/images"
DEFAULT_RESULT_PATH = "test/result"
