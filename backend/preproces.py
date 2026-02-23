import os
from dotenv import load_dotenv
from chonkie import Pipeline, Document
from docling.document_converter import DocumentConverter
import weaviate
from weaviate.classes.config import Configure

from backend.common import PARTIPROGRAM_PATHS

load_dotenv()

converter = DocumentConverter()



OLLAMA_ENDPOINT = os.getenv("WEAVIATE_MODEL_API_ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def collection_exists(collection_name: str) -> bool:
    with weaviate.connect_to_local() as client:
        return client.collections.exists(collection_name)


def create_collection_and_store_chunks(party_id: str, document: Document) -> weaviate.Collection:
    with weaviate.connect_to_local() as client:
        # Create collection for party
        party_collection = client.collections.create(
            name=party_id,
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
        .chunk_with("recursive", chunk_size=500)
        .refine_with("overlap", context_size=100)
    ).run(md_text)

    return document


def process_partiprogram(party_id: str):

    if collection_exists(party_id):
        return

    # Read PDF/html, convert to markdown and extract chunks
    document_raw = converter.convert(PARTIPROGRAM_PATHS[party_id]).document
    md_text = document_raw.export_to_markdown()
    document = process_chunks(md_text)

    # create collection and store chunks
    create_collection_and_store_chunks(party_id, document)