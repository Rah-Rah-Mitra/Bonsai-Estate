"""Deterministic IFC GlobalIds: repeated builds of the same spec write byte-identical files.

ifcopenshell.api calls ``ifcopenshell.guid.new()`` through the module attribute, so swapping that function for
a seeded sequence while a file is being authored makes every GlobalId reproducible.
"""
from __future__ import annotations

import contextlib
import hashlib
import uuid

import ifcopenshell.guid


def from_text(text: str) -> str:
    """A stable 22-character IFC GlobalId for a name (e.g. 'SN5/IfcProject')."""
    return ifcopenshell.guid.compress(uuid.UUID(bytes=hashlib.sha256(text.encode()).digest()[:16]).hex)


class GuidSequence:
    def __init__(self, key: str):
        self.key = key
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return from_text(f"{self.key}/{self.n}")


_CLS = ifcopenshell.ifcopenshell_wrapper.entity_instance
_SWIG_HASH_ATTR = _CLS.__dict__["__hash__"]      # the bound-method descriptor, restored on exit
_SWIG_HASH = _CLS.__hash__                        # raw builtin, used as the fallback inside _stable_hash


def _stable_hash(self):
    """ifcopenshell.api iterates over set()s of entity instances; the native hash follows memory addresses, so
    set order (and with it entity numbering and list order) changes from run to run. Hash by STEP id instead."""
    i = self.id()
    return i if i else _SWIG_HASH(self)


@contextlib.contextmanager
def deterministic(key: str):
    old = ifcopenshell.guid.new
    ifcopenshell.guid.new = GuidSequence(key)
    _CLS.__hash__ = _stable_hash
    try:
        yield ifcopenshell.guid.new
    finally:
        ifcopenshell.guid.new = old
        _CLS.__hash__ = _SWIG_HASH_ATTR
