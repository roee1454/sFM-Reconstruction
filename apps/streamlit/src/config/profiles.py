from enum import Enum
from . import settings

class QualityProfile(Enum):
    SPEED = "SPEED"
    BALANCED = "BALANCED"
    QUALITY = "QUALITY"


OPENMVS_PROFILES = {
    "SPEED": {
        "DensifyPointCloud": {
            "--resolution-level": "3",      # Scale down 3x for speed
            "--max-resolution": "2560",
            "--min-resolution": "640",
            "--number-views": "3",          # Fewer views = faster
            "--iters": "2",                 # Fewer iterations
            "--geometric-iters": "1",       # Minimal geometric passes
            "--sub-resolution-levels": "0", # Disable sub-resolution
            "--remove-dmaps": "1",          # Clean up depth maps
        },
        "ReconstructMesh": {
            "--min-point-distance": "2.0",  # Sparser mesh
            "--decimate": "0.7",            # 30% mesh reduction
            "--remove-spurious": "20",
            "--close-holes": "30",
            "--smooth": "2",
        },
        "RefineMesh": {
            "--resolution-level": "2",      # Lower resolution
            "--max-views": "4",             # Fewer views = faster
            "--scales": "1",                # Single scale
            "--reduce-memory": "1",
        },
        "TextureMesh": {
            "--resolution-level": "1",      # Scale down to avoid OOM
            "--max-texture-size": "4096",   # Smaller texture atlas
            "--cost-smoothness-ratio": "0.1",
            "--global-seam-leveling": "1",
            "--local-seam-leveling": "1",
            "--patch-packing-heuristic": "100",  # Fastest packing
            "--outlier-threshold": "0.02",  # Very low = keep almost all patches
            "--empty-color": "0",           # Black instead of orange for uncovered faces
            "--virtual-face-images": "3",   # Generate texture for coplanar faces sharing views
        },
        "skip_refine_mesh": False,  # Skip RefineMesh for maximum speed
    },
    "BALANCED": {
        "DensifyPointCloud": {
            "--resolution-level": "2",      # Moderate downscale
            "--max-resolution": "2560",
            "--min-resolution": "640",
            "--number-views": "5",          # Balanced views
            "--iters": "3",                 # Standard iterations
            "--geometric-iters": "2",       # Standard geometric passes
            "--sub-resolution-levels": "1", # Light sub-resolution
            "--remove-dmaps": "1",          # Clean up depth maps
        },
        "ReconstructMesh": {
            "--min-point-distance": "1.5",  # Moderate density
            "--decimate": "0.9",            # Light decimation
            "--remove-spurious": "20",
            "--close-holes": "30",
            "--smooth": "2",
        },
        "RefineMesh": {
            "--resolution-level": "1",      # Moderate resolution
            "--max-views": "6",             # Moderate views
            "--scales": "2",                # Dual scale
            "--reduce-memory": "1",
        },
        "TextureMesh": {
            "--resolution-level": "1",      # Scale down to avoid OOM
            "--max-texture-size": "4096",   # Medium texture atlas
            "--cost-smoothness-ratio": "0.1",
            "--global-seam-leveling": "1",
            "--local-seam-leveling": "1",
            "--patch-packing-heuristic": "3",  # Good quality packing
            "--outlier-threshold": "0.03",
            "--empty-color": "0",           # Black instead of orange
            "--virtual-face-images": "3",   # Generate texture for coplanar faces
        },
        "skip_refine_mesh": True,  # Skip RefineMesh for reasonable speed
    },
    "QUALITY": {
        "DensifyPointCloud": {
            "--resolution-level": "1",      # Minimal downscale
            "--max-resolution": "3200",     # Higher max resolution
            "--min-resolution": "640",
            "--number-views": "8",          # More views = better quality
            "--iters": "4",                 # More iterations
            "--geometric-iters": "3",       # More geometric passes
            "--sub-resolution-levels": "2", # Enable sub-resolution
            "--remove-dmaps": "0",          # Keep depth maps for debugging
        },
        "ReconstructMesh": {
            "--min-point-distance": "1.0",  # Denser mesh
            "--decimate": "1",              # No decimation
            "--remove-spurious": "20",
            "--close-holes": "30",
            "--smooth": "2",
        },
        "RefineMesh": {
            "--resolution-level": "0",      # Full resolution
            "--max-views": "12",            # More views
            "--scales": "3",                # Multi-scale
            "--reduce-memory": "1",
        },
        "TextureMesh": {
            "--resolution-level": "0",      # Keep at 0 for quality
            "--max-texture-size": "8192",   # Large texture atlas
            "--cost-smoothness-ratio": "0.1",
            "--global-seam-leveling": "1",
            "--local-seam-leveling": "1",
            "--patch-packing-heuristic": "3",  # Better quality packing
            "--outlier-threshold": "0.03",    # Lower = keep more patches
            "--sharpness-weight": "0.5",       # Apply sharpening
            "--empty-color": "0",              # Black instead of orange
        },
        "skip_refine_mesh": False,  # Run RefineMesh for best quality
    },
}


# COLMAP parameters for each quality profile
COLMAP_PROFILES = {
    "SPEED": {
        "feature_extraction": {
            "max_image_size": 1600,         # Smaller images = faster
            "num_threads": 16,              # Use all cores
            "first_octave": 0,              # Standard SIFT
        },
        "matching": {
            "num_threads": 16,
            "guided_matching": False,       # Faster but less robust
        },
        "incremental_mapping": {
            "num_threads": 16,
            "ba_global_frames_ratio": 1.4,  # Less frequent bundle adjustment
            "multiple_models": False,
        },
    },
    "BALANCED": {
        "feature_extraction": {
            "max_image_size": 2000,         # Moderate size
            "num_threads": 16,
            "first_octave": 0,
        },
        "matching": {
            "num_threads": 16,
            "guided_matching": False,
        },
        "incremental_mapping": {
            "num_threads": 16,
            "ba_global_frames_ratio": 1.2,  # Moderate BA frequency
            "multiple_models": False,
        },
    },
    "QUALITY": {
        "feature_extraction": {
            "max_image_size": 3200,         # Larger images = more features
            "num_threads": 16,
            "first_octave": -1,             # Extra octave for more features
        },
        "matching": {
            "num_threads": 16,
            "guided_matching": True,        # More robust matching
        },
        "incremental_mapping": {
            "num_threads": 16,
            "ba_global_frames_ratio": 1.1,  # Frequent bundle adjustment
            "multiple_models": False,
        },
    },
}


def get_colmap_params(step_name: str) -> dict:
    """Get COLMAP parameters for a specific step based on current profile."""
    # Use settings.QUALITY_PROFILE to allow runtime updates
    current_profile = getattr(settings, "QUALITY_PROFILE", "BALANCED")
    profile = COLMAP_PROFILES.get(current_profile, COLMAP_PROFILES["BALANCED"])
    return profile.get(step_name, {})


def get_profile_params(step_name: str) -> dict:
    """Get parameters for a specific OpenMVS step based on current profile."""
    # Use settings.QUALITY_PROFILE to allow runtime updates
    current_profile = getattr(settings, "QUALITY_PROFILE", "QUALITY")
    profile = OPENMVS_PROFILES.get(current_profile, OPENMVS_PROFILES["QUALITY"])
    return profile.get(step_name, {})


def build_command_with_params(base_cmd: list, step_name: str) -> list:
    """Build command list with profile-specific parameters."""
    params = get_profile_params(step_name)
    cmd = list(base_cmd)
    for param, value in params.items():
        cmd.extend([param, value])
    return cmd


def should_skip_refine_mesh() -> bool:
    """Check if RefineMesh should be skipped based on profile."""
    current_profile = getattr(settings, "QUALITY_PROFILE", "QUALITY")
    profile = OPENMVS_PROFILES.get(current_profile, OPENMVS_PROFILES["QUALITY"])
    return profile.get("skip_refine_mesh", False)
