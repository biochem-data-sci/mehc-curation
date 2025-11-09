# MEHC Curation

A comprehensive Python toolkit for molecular data curation, including validation, cleaning, normalization, and refinement pipelines.

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
from mehc_curation.validation import ValidationStage
from mehc_curation.cleaning import CleaningStage
from mehc_curation.normalization import NormalizationStage
from mehc_curation.refinement import RefinementStage

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
python -m mehc_curation.validation -i input.csv -o output/ -c 5

# Cleaning
python -m mehc_curation.cleaning -i input.csv -o output/ -c 3

# Normalization
python -m mehc_curation.normalization -i input.csv -o output/ -c 3

# Complete refinement
python -m mehc_curation.refinement -i input.csv -o output/ --get_report
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

### Output Directories

- `output_dir` is optional for every stage. If you omit it, data stays in memory and any generated reports are written to the current working directory.
- When you do provide an `output_dir`, the folder will be created automatically if it does not exist, and both CSV outputs and reports are saved beneath it.

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
  title={MEHC-curation: An Automated Python Framework for High-Quality Molecular Dataset Preparation},
  author={Chinh Pham and Nhat-Anh Nguyen-Dang and Thanh-Hoang Nguyen-Vo and Binh P. Nguyen},
  month={dec},
  year={2025},
  version={1.0.3},
  url={https://github.com/biochem-data-sci/mehc-curation},
  license={MIT},
  doi={10.5281/zenodo.17566673}, 
  publisher={Zenodo}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/biochem-data-sci/mehc-curation/issues).



