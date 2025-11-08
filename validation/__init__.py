"""
Validation module for SMILES molecular data.

This module provides functionality for validating SMILES strings and removing
unwanted molecular types (mixtures, inorganics, organometallics).
"""
from .core.pipeline import ValidationPipeline
from .core.engine import ValidationEngine


class ValidationStage(ValidationPipeline):
    """
    Validation stage class - alias for ValidationPipeline.
    
    This class maintains backward compatibility with the original API.
    It inherits from ValidationPipeline and provides the same functionality.
    """
    pass


__all__ = ['ValidationStage']