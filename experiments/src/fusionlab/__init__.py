"""Federated parametric search and selectable fusion over haiku.rag 0.79.0.

Composed on the public API only -- no fork. Supplies the two things the library
does not: a different filter per database, and a choice of fusion mode.
"""

from fusionlab.fusion import Hit, federated_search, fuse_rl, fuse_rrf

__all__ = ["Hit", "federated_search", "fuse_rl", "fuse_rrf"]
