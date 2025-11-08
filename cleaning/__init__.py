"""
Cleaning module for SMILES molecular data.

This module provides functionality for cleaning SMILES strings (salt removal,
neutralization).
"""
from .core.pipeline import CleaningPipeline


class CleaningStage(CleaningPipeline):
    """
    Cleaning stage class - alias for CleaningPipeline.
    
    This class maintains backward compatibility with the original API.
    It inherits from CleaningPipeline and provides the same functionality.
    """
    pass


__all__ = ['CleaningStage']

