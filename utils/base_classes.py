"""Base classes and interfaces for MEHC curation."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseProcessor(ABC):
    """Base class for all data processors (validators, cleaners, normalizers)."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data.
        
        Args:
            data: Input DataFrame containing molecular data
        """
        self.data = data
        self.results = {}
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Process the data and return results.
        
        Returns:
            Dict containing processed data and metadata
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the processing."""
        return {
            'input_count': len(self.data) if self.data is not None else 0,
            'processor_type': self.__class__.__name__
        }


class BaseValidator(BaseProcessor):
    """Base class for all validators."""
    
    @abstractmethod
    def validate(self, **kwargs) -> Dict[str, Any]:
        """
        Validate data and return results.
        
        Returns:
            Dict containing validation results and statistics
        """
        pass
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is valid (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement is_valid_smiles")


class BaseCleaner(BaseProcessor):
    """Base class for all cleaners."""
    
    @abstractmethod
    def clean(self, **kwargs) -> Dict[str, Any]:
        """
        Clean data and return results.
        
        Returns:
            Dict containing cleaned data and cleaning statistics
        """
        pass
    
    def get_cleaning_stats(self) -> Dict[str, int]:
        """Get statistics about the cleaning process."""
        return {
            'items_cleaned': 0,
            'items_removed': 0,
            'items_modified': 0
        }


class BaseNormalizer(BaseProcessor):
    """Base class for all normalizers."""
    
    @abstractmethod
    def normalize(self, **kwargs) -> Dict[str, Any]:
        """
        Normalize data and return results.
        
        Returns:
            Dict containing normalized data and normalization statistics
        """
        pass
    
    def get_normalization_stats(self) -> Dict[str, int]:
        """Get statistics about the normalization process."""
        return {
            'items_normalized': 0,
            'tautomers_standardized': 0,
            'stereochemistry_resolved': 0
        }


class ProcessingPipeline:
    """A pipeline for chaining multiple processors together."""
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.processors = []
        self.results = {}
    
    def add_processor(self, processor: BaseProcessor) -> 'ProcessingPipeline':
        """
        Add a processor to the pipeline.
        
        Args:
            processor: A processor instance
            
        Returns:
            Self for method chaining
        """
        self.processors.append(processor)
        return self
    
    def run(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Run the entire pipeline.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional arguments for processors
            
        Returns:
            Dict containing final results and all intermediate results
        """
        current_data = data
        all_results = {}
        
        for i, processor in enumerate(self.processors):
            processor.data = current_data
            
            # Call the appropriate method based on processor type
            if isinstance(processor, BaseValidator):
                result = processor.validate(**kwargs)
            elif isinstance(processor, BaseCleaner):
                result = processor.clean(**kwargs)
            elif isinstance(processor, BaseNormalizer):
                result = processor.normalize(**kwargs)
            else:
                result = processor.process(**kwargs)
            
            # Update data for next processor if 'data' key exists in result
            if 'data' in result:
                current_data = result['data']
            
            all_results[f'step_{i}_{processor.__class__.__name__}'] = result
        
        all_results['final_data'] = current_data
        self.results = all_results
        return all_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the entire pipeline execution."""
        return {
            'total_processors': len(self.processors),
            'processor_types': [p.__class__.__name__ for p in self.processors],
            'pipeline_completed': len(self.results) > 0
        }