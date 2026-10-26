"""The single boundary between SMILES strings and RDKit molecules.

Every parse and every serialization in the package goes through this module, so
the stages share one conversion instead of each step repeating it.
"""
from typing import Any, Dict, NamedTuple, Optional

import pandas as pd
from rdkit import Chem

from .common import parallel_apply


class MolRecord(NamedTuple):
    """A molecule plus the exact string it was parsed from.

    The string is kept because legacy report counts compare a step's output
    against its input text, not against a re-canonicalised form.
    """

    mol: Optional[Chem.Mol]
    smi: Any


def parse(smi: Any) -> MolRecord:
    """Parse one SMILES. Invalid or non-string input yields ``mol=None``."""
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        mol = None
    return MolRecord(mol, smi)


def parse_series(series, n_cpu: int = -1, split_factor: int = 1):
    """Parse a Series of SMILES into a Series of MolRecord, index preserved."""
    return parallel_apply(series, parse, n_cpu, split_factor)


def to_smiles(mol, isomeric: bool = True) -> str:
    """Serialize a molecule; the only Mol -> SMILES conversion in the package."""
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def changed(record: MolRecord, new_mol) -> bool:
    """True when ``new_mol`` does not serialize back to the record's input string."""
    return to_smiles(new_mol) != record.smi


def records_for(series, mol_cache: Optional[Dict[Any, MolRecord]] = None,
                n_cpu: int = -1, split_factor: int = 1):
    """Records for a column, reusing cache entries that still match its string."""
    if not mol_cache:
        return parse_series(series, n_cpu, split_factor)

    items = list(series.items())
    records = [None] * len(items)
    misses, miss_pos = [], []
    for pos, (idx, smi) in enumerate(items):
        cached = mol_cache.get(idx)
        if cached is not None and cached.smi == smi:
            records[pos] = cached
        else:
            misses.append(smi)
            miss_pos.append(pos)

    if misses:
        parsed = parse_series(pd.Series(misses), n_cpu, split_factor).tolist()
        for pos, record in zip(miss_pos, parsed):
            records[pos] = record

    return pd.Series(records, index=series.index, dtype=object)


def prune(mol_cache: Optional[Dict[Any, MolRecord]], index) -> None:
    """Drop cache entries whose row is gone; misses only cost a re-parse."""
    if not mol_cache:
        return
    keep = set(index)
    for key in [key for key in mol_cache if key not in keep]:
        del mol_cache[key]
