"""
ShakeNBreak is a defect structure-searching method employing chemically-guided
bond distortions to locate ground-state and metastable structures of point
defects in solid materials.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("shakenbreak")  # from package metadata (pyproject.toml)
except importlib.metadata.PackageNotFoundError:
    __version__ = "No version found"  # fallback for local development or if package isn't installed
