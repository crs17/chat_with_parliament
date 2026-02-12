import os
from dotenv import load_dotenv
from chonkie import Pipeline, Document
from docling.document_converter import DocumentConverter
import httpx
from pathlib import Path
import weaviate
from weaviate.classes.config import Configure


load_dotenv()

converter = DocumentConverter()

partiprogram_urls = {
    'A': 'https://www.socialdemokratiet.dk/media/ws1fxjky/socialdemokratiets-principprogram-faelles-om-danmark.pdf',
}

def create_collection_if_not_exists(party_id: str) -> weaviate.Collection:
    with weaviate.connect_to_local() as client:

        if client.collections.exists(party_id):
            print(f"Collection {party_id} already exists")
            return False
        
        party_collection = client.collections.create(
            name=party_id,
            vector_config=Configure.Vectors.text2vec_ollama(
                #api_endpoint='http://host.docker.internal:11434',
                api_endpoint=os.getenv('MODEL_API_ENDPOINT'),
                model='nomic-embed-text'
            )
        )
    return True


def chunk_pdf(path: Path) -> Document:

    return process_chunks(md_text)


def process_partiprogram(party_id: str) -> Document:

    if not create_collection_if_not_exists(party_id):
        return

    url = partiprogram_urls[party_id]
    document_raw = converter.convert(url).document
    md_text = document_raw.export_to_markdown()

    return process_chunks(md_text)




def process_chunks(md_text: str) -> Document:
    document = (Pipeline()
        .process_with("markdown")
        .chunk_with("recursive", chunk_size=500)
        .refine_with("contextual_header")
        .refine_with("overlap", context_size=100)
        .store_in("weaviate", collection_name="A")
    ).run(md_text)

    return document

def main():
    for party_id in partiprogram_urls:
        process_partiprogram(party_id)
    
if __name__ == "__main__":
    main()