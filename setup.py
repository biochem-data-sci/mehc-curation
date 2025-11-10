"""Setup script for mehc_curation package."""
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MEHC-Curation: A Python Framework for High-Quality Molecular Dataset Curation"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='mehc-curation',
    version='1.0.5',
    author='Thanh-Hoang Nguyen-Vo',
    author_email='nvthoang@gmail.com',
    description='A comprehensive toolkit for molecular data curation, validation, cleaning, and normalization',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/biochem-data-sci/mehc-curation',
    packages=find_packages(include=["mehc_curation", "mehc_curation.*"]),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=read_requirements(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mehc-validation=mehc_curation.validation.__main__:main',
            'mehc-cleaning=mehc_curation.cleaning.__main__:main',
            'mehc-normalization=mehc_curation.normalization.__main__:main',
            'mehc-refinement=mehc_curation.refinement.__main__:main',
        ],
    },
    keywords='chemistry, SMILES, molecular data, validation, cleaning, normalization, cheminformatics, RDKit',
    project_urls={
        'Bug Reports': 'https://github.com/biochem-data-sci/mehc-curation/issues',
        'Source': 'https://github.com/biochem-data-sci/mehc-curation',
        'Documentation': 'https://github.com/biochem-data-sci/mehc-curation#readme',
    },
)

