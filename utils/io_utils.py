"""File I/O operations."""
import os
import pandas as pd
from pathlib import Path
from typing import Optional


def setup_data_dir() -> str:
    """
    Set up and return the path to the data directory.
    
    Returns:
        str: Path to the data directory containing reference files (metals.txt, etc.)
    """
    # Get the package root directory (mehc_curation/)
    package_root = Path(__file__).parent.parent
    
    # Data directory at package level - contains metals.txt etc.
    dat_dir = package_root / "dat"
    
    # Create directory if it doesn't exist
    dat_dir.mkdir(exist_ok=True)
    
    return str(dat_dir)


def setup_template_dir() -> str:
    """
    Set up and return the path to the template directory.
    
    Returns:
        str: Path to the template directory containing report templates
    """
    # Get the package root directory (mehc_curation/)
    package_root = Path(__file__).parent.parent
    
    # Template directory at package level - contains validity_check.txt, deduplicate.txt, end.txt
    template_dir = package_root / "template_report"
    
    return str(template_dir)


def read_csv_safely(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame or None if read fails
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' does not exist.")
            return None
        
        df = pd.read_csv(filepath, **kwargs)
        if df.empty:
            print(f"Warning: File '{filepath}' is empty.")
            return None
        return df
        
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None


def save_csv_safely(df: pd.DataFrame, filepath: str, **kwargs) -> bool:
    """
    Safely save a DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        **kwargs: Additional arguments for to_csv
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, **kwargs)
        return True
    except Exception as e:
        print(f"Error saving file '{filepath}': {e}")
        return False


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_extension(filepath: str) -> str:
    """
    Get the file extension from a filepath.
    
    Args:
        filepath: Path to file
        
    Returns:
        str: File extension including the dot (e.g., '.csv', '.txt')
    """
    return os.path.splitext(filepath)[1]