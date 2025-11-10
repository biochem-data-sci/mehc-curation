# Installation Guide

## Prerequisites

Before installing `mehc-curation`, you need to install RDKit, which is a core dependency for molecular processing.

### Installing RDKit (Required)

RDKit is best installed via conda, even if you're using pip for other packages:

```bash
# Create a new conda environment (recommended)
conda create -n mehc_env python=3.9
conda activate mehc_env

# Install RDKit
conda install -c conda-forge rdkit
```

### Alternative: Using pip for RDKit

If you cannot use conda, you can try installing RDKit via pip:

```bash
pip install rdkit-pypi
```

**Note**: RDKit via pip may not work on all platforms. Conda installation is strongly recommended.

## Installing mehc-curation

### Option 1: Install from PyPI (Recommended)

```bash
# Make sure RDKit is installed first (see above)
pip install -U mehc-curation==1.0.5
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/biochem-data-sci/mehc-curation.git
cd mehc-curation

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Option 3: Install with conda

```bash
# Create environment with RDKit
conda create -n mehc_env python=3.9 rdkit -c conda-forge
conda activate mehc_env

# Install mehc-curation
pip install mehc-curation
```

## Verifying Installation

After installation, you can run a quick smoke test to confirm imports and optional output handling:

```python
import pandas as pd
from mehc_curation.validation import ValidationStage
from mehc_curation.utils.common import deduplicate

# Minimal DataFrame
df = pd.DataFrame({'smiles': ['CCO', 'CC']})
validator = ValidationStage(df)

# Deduplication without an output directory (keeps data in memory)
cleaned = deduplicate(df, output_dir=None, print_logs=False)

# Deduplication with automatic directory creation
cleaned_with_reports, _ = deduplicate(
    df,
    output_dir="mehc_outputs/demo",
    print_logs=False,
    get_report=True,
    return_format_data=True,
)

print("Installation successful!")
```

## Troubleshooting

### RDKit Import Error

If you see `ModuleNotFoundError: No module named 'rdkit'`:

1. Make sure RDKit is installed:
   ```bash
   conda install -c conda-forge rdkit
   ```

2. Verify RDKit installation:
   ```python
   import rdkit
   print(rdkit.__version__)
   ```

### Parallel Pandas Issues

If you encounter issues with parallel processing:

1. Make sure parallel-pandas is installed:
   ```bash
   pip install parallel-pandas
   ```

2. Check your Python version (requires Python >= 3.7)

## System Requirements

- Python >= 3.7
- pandas >= 1.3.0
- parallel-pandas >= 0.2.8
- RDKit (install via conda: `conda install -c conda-forge rdkit`)

## Development Installation

For development, install with development dependencies:

```bash
git clone https://github.com/biochem-data-sci/mehc-curation.git
cd mehc-curation
pip install -e ".[dev]"
```

