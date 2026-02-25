import os
import tempfile
from dotenv import load_dotenv
from chonkie import Pipeline, Document
from docling.document_converter import DocumentConverter
from playwright.sync_api import sync_playwright
import weaviate
from weaviate.classes.config import Configure

from backend.common import PARTY_MANIFESTS, weaviate_collection_name

load_dotenv()

converter = DocumentConverter()


def _fetch_url_with_browser(url: str, page) -> str:
    """Load URL in a real browser (passes JS/cookie validation); return path to temp HTML file for docling."""
    # Use "load" (not "networkidle") so we don't timeout on sites with persistent connections (analytics, etc.)
    page.goto(url, wait_until="load", timeout=30_000)
    html = page.content()
    fd, path = tempfile.mkstemp(suffix=".html")
    try:
        os.write(fd, html.encode("utf-8"))
    finally:
        os.close(fd)
    return path



OLLAMA_ENDPOINT = os.getenv("WEAVIATE_MODEL_API_ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def collection_exists(party_id: str) -> bool:
    with weaviate.connect_to_local() as client:
        return client.collections.exists(weaviate_collection_name(party_id))


def create_collection_and_store_chunks(party_id: str, document: Document) -> weaviate.Collection:
    with weaviate.connect_to_local() as client:
        # Create collection for party
        party_collection = client.collections.create(
            name=weaviate_collection_name(party_id),
            vector_config=Configure.Vectors.text2vec_ollama(
                api_endpoint=OLLAMA_ENDPOINT,
                model=EMBEDDING_MODEL,
            ),
        )

        # store chunks in newly created collection
        with party_collection.batch.fixed_size(batch_size=100) as batch:
            for chunk in document.chunks:
                batch.add_object(properties={"text": chunk.text})

            if batch.number_errors:
                print(f"Batch had {batch.number_errors} errors")
                for obj in party_collection.batch.failed_objects:
                    print("Failed:", obj)
            else:
                print(f"Inserted {len(document.chunks)} chunks into {party_id}")

    return party_collection


def process_chunks(md_text: str) -> Document:
    document = (Pipeline()
        .process_with("markdown")
        .chunk_with("recursive", chunk_size=1000)
        .refine_with("overlap", context_size=100)
    ).run(md_text)

    return document


def process_partiprogram(party_id: str):
    if collection_exists(party_id):
        return

    if party_id not in PARTY_MANIFESTS:
        raise ValueError(f"No manifest URLs for party {party_id!r}")

    md_parts: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            for url in PARTY_MANIFESTS[party_id]:
                print(f"Fetching: {url}")
                path = _fetch_url_with_browser(url, page)
                try:
                    result = converter.convert(path)
                    md_parts.append(result.document.export_to_markdown())
                finally:
                    os.unlink(path)
        finally:
            browser.close()
    md_text = "\n\n".join(md_parts)
    document = process_chunks(md_text)

    create_collection_and_store_chunks(party_id, document)