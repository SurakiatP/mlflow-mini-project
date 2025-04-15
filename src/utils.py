from typing import List
import numpy as np

def validate_input(features: List[float]) -> bool:
    """Check that input is a list of 4 numeric values (iris features)."""
    if not isinstance(features, list):
        return False
    if len(features) != 4:
        return False
    return all(isinstance(x, (int, float)) for x in features)

def prepare_input(features: List[float]):
    """Convert validated features to model input format (2D array)."""
    return np.array([features])
