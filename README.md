# MEHC-CURATION
_This tool is developed by Chinh Pham Trong, Huy-Quang Trinh and Thanh-Hoang Nguyen Vo*._

## 1. General

MEHC-curation is a tool designed to enhance the quality of molecular datasets through systematic curation. 
By refining the input data, it helps ensure more reliable and accurate results when training computational models, 
such as those used in QSAR (Quantitative Structure-Activity Relationship) studies.

### 1.1 Prerequisite

To run this project, you need the Python version ≥ 3.10, following Python libraries:

- [`rdkit (2023.03.3)`](https://www.rdkit.org/) – cheminformatics and molecule handling  
- [`pandas (2.2.3)`](https://pandas.pydata.org/) – data manipulation and I/O  
- [`numpy (2.0.2)`](https://numpy.org/) – array and math operations
- [`parallel-pandas (0.6.4)`](https://pypi.org/project/parallel-pandas/) – parallelized DataFrame operations

### 1.2 Installation
#### Interactive on CLI

```bash
conda create -n smiles-curation python=3.12 -y
conda activate smiles-curation
```

```bash
conda install -c rdkit rdkit=2023.03.3
```

```bash
conda install pandas=2.2.3 numpy=2.0.2
```

```bash
pip install parallel-pandas==0.6.4
```

### 1.3 Input and output format

- Input: pd.DataFrame with the SMILES data in the first column. The dataset can be imported in several file format (e.g. csv, parquet, etc.)
- Output: pd.DataFrame with the curated and standardized SMILES data.

## 2. Recommend way for running MEHC-Curation
### 2.1 Running _validate.py_
_validate.py_ is used to validate chemical datasets as well as remove mixtures, inorganics and organometallics.

An example command for this module is shown below:

```bash
python ./curation/validate.py -i ./data/smiles_df.csv -o ./reports -c 5
```

#### Argument reference

| Argument               | Alias | Type   | Description                                                   | Default | Required |
|------------------------|-------|--------|---------------------------------------------------------------|---------|----------|
| `--input`              | `-i`  | `str`  | Path to the input SMILES CSV file                             | —       | ✅        |
| `--output`             | `-o`  | `str`  | Output folder path. Will include CSV and report (if selected) | —       | ✅        |
| `--choice`             | `-c`  | `int`  | Action to perform (1–5)                                       | —       | ✅        |
| `--validate_beginning` | —     | `flag` | Validate SMILES before any action (skipped for choice 1)      | `False` | ❌        |
| `--deduplicate`        | —     | `flag` | Remove duplicate SMILES after the action                      | `False` | ❌        |
| `--get_report`         | —     | `flag` | Generate a report and summary CSV                             | `False` | ❌        |
| `--print_logs`         | `-p`  | `flag` | Print logs to terminal (use `--print_logs false` to disable)  | `True`  | ❌        |
| `--n_cpu`              | —     | `int`  | Number of CPUs to use                                         | `1`     | ❌        |
| `--split_factor`       | —     | `int`  | Data split factor for processing                              | `1`     | ❌        |

### 2.2 Running _cleaning.py_
_cleaning.py_ is used to cleaning salts and neutralizing chemical datasets.

An example command for this module is shown below:

```bash
python ./curation/cleaning.py -i ./data/smiles_df.csv -o ./reports -c 3
```

#### Argument reference

| Argument               | Alias | Type   | Description                                                   | Default | Required |
|------------------------|-------|--------|---------------------------------------------------------------|---------|----------|
| `--input`              | `-i`  | `str`  | Path to the input SMILES CSV file                             | —       | ✅        |
| `--output`             | `-o`  | `str`  | Output folder path. Will include CSV and report (if selected) | —       | ✅        |
| `--choice`             | `-c`  | `int`  | Action to perform (1–3)                                       | —       | ✅        |
| `--validate_beginning` | —     | `flag` | Validate SMILES before any action                             | `False` | ❌        |
| `--deduplicate`        | —     | `flag` | Remove duplicate SMILES after the action                      | `False` | ❌        |
| `--get_report`         | —     | `flag` | Generate a report and summary CSV                             | `False` | ❌        |
| `--print_logs`         | `-p`  | `flag` | Print logs to terminal (use `--print_logs false` to disable)  | `True`  | ❌        |
| `--n_cpu`              | —     | `int`  | Number of CPUs to use                                         | `1`     | ❌        |
| `--split_factor`       | —     | `int`  | Data split factor for processing                              | `1`     | ❌        |

### 2.3 Running _normalization.py_
_normalization.py_ is used to detautomerize and destereoisomerize chemical datasets.

An example command for this module is shown below:

```bash
python ./curation/normalization.py -i ./data/smiles_df.csv -o ./reports -c 3
```

#### Argument reference

| Argument               | Alias | Type   | Description                                                   | Default | Required |
|------------------------|-------|--------|---------------------------------------------------------------|---------|----------|
| `--input`              | `-i`  | `str`  | Path to the input SMILES CSV file                             | —       | ✅        |
| `--output`             | `-o`  | `str`  | Output folder path. Will include CSV and report (if selected) | —       | ✅        |
| `--choice`             | `-c`  | `int`  | Action to perform (1–3)                                       | —       | ✅        |
| `--validate_beginning` | —     | `flag` | Validate SMILES before any action                             | `False` | ❌        |
| `--deduplicate`        | —     | `flag` | Remove duplicate SMILES after the action                      | `False` | ❌        |
| `--get_report`         | —     | `flag` | Generate a report and summary CSV                             | `False` | ❌        |
| `--print_logs`         | `-p`  | `flag` | Print logs to terminal (use `--print_logs false` to disable)  | `True`  | ❌        |
| `--n_cpu`              | —     | `int`  | Number of CPUs to use                                         | `1`     | ❌        |
| `--split_factor`       | —     | `int`  | Data split factor for processing                              | `1`     | ❌        |

### 2.4 Running _refinement.py_
_refinement.py_ is a module that performs an automatically end-to-end pipeline to curate chemical datasets.

```bash
python ./curation/refinement.py -i ./data/smiles_df.csv -o ./reports
```

#### Argument reference

| Argument               | Alias | Type   | Description                                                   | Default | Required |
|------------------------|-------|--------|---------------------------------------------------------------|---------|----------|
| `--input`              | `-i`  | `str`  | Path to the input SMILES CSV file                             | —       | ✅        |
| `--output`             | `-o`  | `str`  | Output folder path. Will include CSV and report (if selected) | —       | ✅        |
| `--validate_beginning` | —     | `flag` | Validate SMILES before any action                             | `True`  | ❌        |
| `--rm_mixtures`        | —     | `flag` | Remove mixtures from SMILES data                              | `True`  | ❌        |
| `--rm_inorganics`      | —     | `flag` | Remove inorganic molecules from SMILES data                   | `True`  | ❌        |
| `--rm_organometallics` | —     | `flag` | Remove organometallics from SMILES data                       | `True`  | ❌        |
| `--cl_salts`           | —     | `flag` | Remove salts from SMILES data                                 | `True`  | ❌        |
| `--neutralize`         | —     | `flag` | Neutralize charged molecules from SMILES data                 | `True`  | ❌        |
| `--destereoisomerize`  | —     | `flag` | Perform destereoisomerization                                 | `True`  | ❌        |
| `--detautomerize`      | —     | `flag` | Perform detautomerization                                     | `True`  | ❌        |
| `--deduplicate`        | —     | `flag` | Remove duplicate SMILES after each stage                      | `True`  | ❌        |
| `--print_logs`         | `-p`  | `flag` | Print logs to terminal (use `--print_logs false` to disable)  | `True`  | ❌        |
| `--get_report`         | —     | `flag` | Generate a report and summary CSV                             | `False` | ❌        |
| `--n_cpu`              | —     | `int`  | Number of CPUs to use                                         | `1`     | ❌        |
| `--split_factor`       | —     | `int`  | Data split factor for processing                              | `1`     | ❌        |