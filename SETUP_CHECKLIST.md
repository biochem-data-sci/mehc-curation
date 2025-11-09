# Pre-Publication Checklist

Before publishing to PyPI, complete the following:

## Required Updates

- [ ] **Update author information** in:
  - `setup.py` (lines 24-25)
  - `pyproject.toml` (lines 10-11)

- [x] **Update repository URLs** - ✅ Completed: `https://github.com/biochem-data-sci/mehc-curation`

- [ ] **Set initial version** (currently 1.0.0):
  - `setup.py` (line 23)
  - `pyproject.toml` (line 5)

- [ ] **Add LICENSE file** (MIT License already created)

- [ ] **Review README.md** and update:
  - Installation instructions
  - Usage examples
  - Citation information

## Files Created

✅ `setup.py` - Main setup script
✅ `pyproject.toml` - Modern Python packaging configuration
✅ `requirements.txt` - Python dependencies
✅ `MANIFEST.in` - Files to include in distribution
✅ `README.md` - Package documentation
✅ `LICENSE` - MIT License
✅ `.gitignore` - Git ignore rules
✅ `INSTALL.md` - Detailed installation guide
✅ `PUBLISHING.md` - Publishing instructions

## Testing Before Publication

1. **Test local installation**:
   ```bash
   pip install -e .
   ```

2. **Test imports**:
   ```python
   from mehc_curation.validation import ValidationStage
   from mehc_curation.cleaning import CleaningStage
   from mehc_curation.normalization import NormalizationStage
   from mehc_curation.refinement import RefinementStage
   ```

3. **Test CLI commands**:
   ```bash
   mehc-validation --help
   mehc-cleaning --help
   mehc-normalization --help
   mehc-refinement --help
   ```

4. **Build distribution**:
   ```bash
   python -m build
   ```

5. **Check distribution contents**:
   ```bash
   tar -tzf dist/mehc-curation-1.0.0.tar.gz | head -20
   ```

## Important Notes

- **RDKit dependency**: RDKit is NOT included in `requirements.txt` because it should be installed via conda. Users must install it separately.
- **Package structure**: The package uses `find_packages()` to automatically discover all subpackages under `mehc_curation` (validation, cleaning, normalization, refinement, utils).
- **Data files**: Template and data files are included via `MANIFEST.in`.

## Next Steps

1. Complete all checklist items above
2. Test locally
3. Follow `PUBLISHING.md` guide to publish to PyPI
4. Update documentation with actual PyPI installation command

