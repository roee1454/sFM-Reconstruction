#!/usr/bin/env python3
"""
Visualization entry point for the ObjectReconstruction project.
"""

from src.config.settings import DEFAULT_RESULT_PATH
from src.visualization import visualize_mesh
import os

def main():
    result_path = os.path.join(DEFAULT_RESULT_PATH, "result.obj")
    visualize_mesh(result_path)

if __name__ == "__main__":
    main()