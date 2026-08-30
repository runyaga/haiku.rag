"""The pure parts of `build.py`.

`build.py` is omitted from the coverage gate because three of its lines call
LanceDB. The rest is ordinary logic and is tested here, so the omission covers
only the IO it was claimed to cover.
"""

import json

import pytest

from fusionlab import corpus as c
from fusionlab.build import _imports, _manifest_is_current


class TestImports:
    def test_one_import_per_document(self):
        docs = c.build_corpus(docs_per_db=2)
        assert len(_imports(docs)) == len(docs)

    def test_each_carries_its_uri_title_and_metadata(self):
        doc = c.build_corpus(docs_per_db=1)[0]
        prepared = _imports([doc])[0]
        assert prepared.uri == doc.uri
        assert prepared.title == doc.title
        assert prepared.metadata == doc.metadata

    def test_each_carries_exactly_one_chunk_holding_the_text(self):
        doc = c.build_corpus(docs_per_db=1)[0]
        [chunk] = _imports([doc])[0].chunks
        assert chunk.content == doc.text
        assert chunk.metadata == {"doc_item_refs": ["#/texts/0"]}

    def test_the_chunk_carries_no_embedding_so_the_store_computes_it(self):
        [chunk] = _imports(c.build_corpus(docs_per_db=1))[0].chunks
        assert chunk.embedding is None

    def test_no_documents_gives_no_imports(self):
        assert _imports([]) == []


class TestManifestShortCircuit:
    def test_a_missing_manifest_is_not_current(self, tmp_path):
        assert not _manifest_is_current(tmp_path / "absent.json", "abc")

    def test_a_matching_manifest_is_current(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest": "abc", "documents": 3}))
        assert _manifest_is_current(path, "abc")

    def test_a_differing_manifest_is_not_current(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest": "abc"}))
        assert not _manifest_is_current(path, "xyz")

    def test_a_manifest_without_the_key_is_not_current(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"documents": 3}))
        assert not _manifest_is_current(path, "abc")

    @pytest.mark.parametrize("junk", ["", "not json", "[]"])
    def test_an_unreadable_manifest_is_not_current(self, tmp_path, junk):
        """A corrupt manifest must force a rebuild, not crash the build."""
        path = tmp_path / "m.json"
        path.write_text(junk)
        assert not _manifest_is_current(path, "abc")
