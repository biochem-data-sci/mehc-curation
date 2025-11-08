"""
Refinement module for SMILES molecular data.

This module orchestrates validation, cleaning, and normalization stages
to provide a complete refinement pipeline.
"""
from .core.pipeline import RefinementPipeline


class RefinementStage(RefinementPipeline):
    """
    Refinement stage class - alias for RefinementPipeline.
    
    This class maintains backward compatibility with the original API.
    It inherits from RefinementPipeline and provides the same functionality.
    """
    pass


__all__ = ['RefinementStage']

