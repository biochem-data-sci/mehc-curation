"""
Normalization module for SMILES molecular data.

This module provides functionality for normalizing SMILES strings (tautomer
and stereoisomer normalization).
"""
from .core.pipeline import NormalizationPipeline


class NormalizationStage(NormalizationPipeline):
    """
    Normalization stage class - alias for NormalizationPipeline.
    
    This class maintains backward compatibility with the original API.
    It inherits from NormalizationPipeline and provides the same functionality.
    """
    pass


__all__ = ['NormalizationStage']

