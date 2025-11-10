# Publishing Guide for mehc-curation

This guide explains how to publish the `mehc-curation` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on:
   - Test PyPI: https://test.pypi.org/
   - PyPI: https://pypi.org/

2. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Step 1: Update Version and Metadata

Before publishing, update:

1. **Version number** in:
   - `setup.py` (line 23: `version='1.0.0'`)
   - `pyproject.toml` (line 5: `version = "1.0.0"`)

2. **Author information** in:
   - `setup.py` (lines 24-25)
   - `pyproject.toml` (lines 10-11)

3. **Repository URLs** - Already updated to: `https://github.com/biochem-data-sci/mehc-curation`

## Step 2: Prepare Distribution Files

### Clean previous builds:
```bash
rm -rf build/ dist/ *.egg-info/
```

### Build the package:
```bash
python -m build
```

This creates:
- `dist/mehc-curation-1.0.0.tar.gz` (source distribution)
- `dist/mehc_curation-1.0.0-py3-none-any.whl` (wheel distribution)

## Step 3: Test on Test PyPI

### Upload to Test PyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your Test PyPI API token

### Test installation from Test PyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ mehc-curation
```

## Step 4: Publish to PyPI

Once testing is successful:

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

## Step 5: Verify Publication

Check your package on PyPI:
- https://pypi.org/project/mehc-curation/

Test installation:
```bash
pip install mehc-curation
```

## Creating API Tokens

1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Create a new token with scope: "Entire account" or "Project: mehc-curation"
4. Copy the token (you'll only see it once!)

## Version Bumping

For future releases, follow semantic versioning:
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.4 → 1.0.5): Bug fixes

Update version in both `setup.py` and `pyproject.toml`, then rebuild and upload.

## Troubleshooting

### "File already exists" error
- Version number already exists on PyPI
- Increment version number and rebuild

### "Invalid distribution" error
- Check that all required files are included
- Verify `MANIFEST.in` includes all necessary files

### Import errors after installation
- Verify package structure is correct
- Check that `__init__.py` files exist in all packages
- Test locally with `pip install -e .`

## Notes

- RDKit is not included in `requirements.txt` because it should be installed via conda
- Users must install RDKit separately: `conda install -c conda-forge rdkit`
- See `INSTALL.md` for detailed installation instructions

