"""Write the synthetic corpora. The only module that needs a database.

Covered by ``tests/test_build_integration.py`` rather than by the coverage gate:
it exists to call LanceDB, and stubbing those writes would test the stub.

Idempotent -- a matching manifest is a no-op, a mismatch rebuilds rather than
appending to a half-written database.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from fusionlab.corpus import DATABASES, SynthDoc, build_corpus, manifest_hash, to_truth

HERE = Path(__file__).resolve().parent.parent.parent / "synth"
TRUTH = HERE / "truth.json"
MANIFEST = HERE / "manifest.json"


def _imports(docs: list[SynthDoc]) -> list[Any]:
    from haiku.rag.client.documents import DocumentImport
    from haiku.rag.store.models import Chunk

    prepared = []
    for doc in docs:
        docling = DoclingDocument(name=doc.uri)
        docling.add_text(label=DocItemLabel.TEXT, text=doc.text)
        prepared.append(
            DocumentImport(
                docling_document=docling,
                chunks=[
                    Chunk(
                        content=doc.text,
                        order=0,
                        metadata={"doc_item_refs": ["#/texts/0"]},
                    )
                ],
                uri=doc.uri,
                title=doc.title,
                metadata=doc.metadata,
            )
        )
    return prepared


def _manifest_is_current(path: Path, manifest: str) -> bool:
    """Whether a recorded manifest matches, tolerating absence and corruption.

    A corrupt manifest must force a rebuild rather than crash: the recorded
    hash is a claim about the corpus, and an unreadable claim is no claim.
    """
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("manifest") == manifest
    except (json.JSONDecodeError, AttributeError):
        return False


async def build(force: bool = False, batch: int = 100) -> str:
    """Build every database. Returns the manifest hash."""
    from haiku.rag.client import HaikuRAG

    docs = build_corpus()
    manifest = manifest_hash(docs)

    if not force:
        if _manifest_is_current(MANIFEST, manifest):
            print(f"manifest {manifest[:16]}… unchanged; nothing to do")
            return manifest
        print("manifest absent or changed; rebuilding from scratch")

    for name in DATABASES:
        path = HERE / f"{name}.lancedb"
        if path.exists():
            shutil.rmtree(path)
        subset = [d for d in docs if d.db == name]
        async with HaikuRAG(create=True, sources=[name]) as rag:
            for start in range(0, len(subset), batch):
                await rag.import_documents(_imports(subset[start : start + batch]))
        print(f"  built {name}: {len(subset)} documents")

    TRUTH.write_text(json.dumps(to_truth(docs), indent=1))
    MANIFEST.write_text(
        json.dumps({"manifest": manifest, "documents": len(docs)}, indent=1)
    )
    print(f"manifest {manifest[:16]}…  truth.json {len(docs)} rows")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(build())
