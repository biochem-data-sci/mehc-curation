# MEHC Curation

A comprehensive Python toolkit for SMILES molecular data curation, including validation, cleaning, normalization, and refinement pipelines.

## Features

- **Validation**: Validate SMILES strings and remove unwanted molecular types (mixtures, inorganics, organometallics)
- **Cleaning**: Remove salts and neutralize charged molecules
- **Normalization**: Normalize tautomers and stereoisomers
- **Refinement**: Complete pipeline orchestrating all stages
- **Parallel Processing**: Efficient parallel processing using all available CPUs by default
- **Comprehensive Reporting**: Generate detailed reports for each processing stage

## Installation

### Prerequisites

Before installing `mehc-curation`, you need to install RDKit, which is best installed via conda:

```bash
conda install -c conda-forge rdkit
```

### Install from PyPI

```bash
pip install mehc-curation
```

### Install from source

```bash
git clone https://github.com/biochem-data-sci/mehc-curation.git
cd mehc-curation
pip install -e .
```

## Quick Start

### Python API

```python
import pandas as pd
from validation import ValidationStage
from cleaning import CleaningStage
from normalization import NormalizationStage
from refinement import RefinementStage

# Load your SMILES data
df = pd.read_csv("your_data.csv")

# Validation
validator = ValidationStage(df)
validated_df = validator.complete_validation()

# Cleaning
cleaner = CleaningStage(validated_df)
cleaned_df = cleaner.complete_cleaning()

# Normalization
normalizer = NormalizationStage(cleaned_df)
normalized_df = normalizer.complete_normalization()

# Complete refinement pipeline
refiner = RefinementStage(df)
refined_df = refiner.complete_refinement(
    output_dir="./output",
    get_report=True
)
```

### Command Line Interface

```bash
# Validation
python -m validation -i input.csv -o output/ -c 5

# Cleaning
python -m cleaning -i input.csv -o output/ -c 3

# Normalization
python -m normalization -i input.csv -o output/ -c 3

# Complete refinement
python -m refinement -i input.csv -o output/ --get_report
```

## Modules

### Validation Module

Validates SMILES strings and removes unwanted molecular types:

- `validate_smi()`: Validate SMILES strings
- `rm_mixture()`: Remove mixture compounds
- `rm_inorganic()`: Remove inorganic compounds
- `rm_organometallic()`: Remove organometallic compounds
- `complete_validation()`: Run all validation steps

### Cleaning Module

Cleans SMILES strings:

- `cl_salt()`: Remove salts from SMILES
- `neutralize()`: Neutralize charged molecules
- `complete_cleaning()`: Run all cleaning steps

### Normalization Module

Normalizes SMILES strings:

- `detautomerize()`: Normalize tautomers
- `destereoisomerize()`: Remove stereoisomers
- `complete_normalization()`: Run all normalization steps

### Refinement Module

Complete refinement pipeline:

- `complete_refinement()`: Orchestrates validation, cleaning, and normalization stages

## Configuration

### CPU Usage

By default, the library uses all available CPUs (`n_cpu=-1`). You can specify the number of CPUs:

```python
# Use all CPUs (default)
refiner.complete_refinement(n_cpu=-1)

# Use specific number of CPUs
refiner.complete_refinement(n_cpu=4)

# Use single CPU
refiner.complete_refinement(n_cpu=1)
```

## Requirements

- Python >= 3.7
- pandas >= 1.3.0
- parallel-pandas >= 0.2.8
- RDKit (install via conda: `conda install -c conda-forge rdkit`)

## License

MIT License - see LICENSE file for details

## Citation

If you use this library in your research, please cite:

```bibtex
@software{mehc_curation,
  title={MEHC Curation: A Comprehensive Toolkit for SMILES Molecular Data Curation},
  author={Your Name},
  year={2024},
  url={https://github.com/biochem-data-sci/mehc-curation}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/biochem-data-sci/mehc-curation/issues).

