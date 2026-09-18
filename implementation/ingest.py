from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer
import pandas as pd

load_dotenv(override=True)

MODEL = "gemini/gemini-2.5-flash"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
PREPROCESSED_DIR = Path(__file__).parent.parent / "preprocessedData"
PREPROCESSED_DIR.mkdir(exist_ok=True)

collection_name = "docs"
embedding_model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 200
wait = wait_exponential(multiplier=1, min=10, max=240)

WORKERS = 3

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk")
    summary: str = Field(description="A few sentences summarizing the content")
    original_text: str = Field(description="The original text of this chunk")

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

def preprocess_csv_and_fetch():
    documents = []

    csv_file = KNOWLEDGE_BASE_PATH / "WWE_History_1000.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file).dropna()
        preprocessed_text = "# Preprocessed WWE Matches\n\n"
        
        for _, row in df.iterrows():
            preprocessed_text += (
                f"Match ID: {row.get('Match ID', '')} | "
                f"Date: {row.get('Date', '')} | "
                f"Event: {row.get('Event', '')} | "
                f"Winner: {row.get('Winner', '')} | "
                f"Loser: {row.get('Loser', '')} | "
                f"Title Match: {row.get('Title Match', '')}\n"
            )
            
        output_file = PREPROCESSED_DIR / "cleaned_matches.md"
        
        if not output_file.exists():
            output_file.write_text(preprocessed_text, encoding="utf-8")
            print(f"Preprocessed data saved to: {output_file}")
        else:
            print(f"Using existing preprocessed file: {output_file}")
            preprocessed_text = output_file.read_text(encoding="utf-8")
            
        documents.append({
            "type": "csv_matches",
            "source": output_file.as_posix(),
            "text": preprocessed_text
        })

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if folder.is_dir():
            doc_type = folder.name
            for file in folder.rglob("*.md"):
                with open(file, "r", encoding="utf-8") as f:
                    documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents total")
    return documents

def make_prompt(document):
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.
Document type: {document["type"]}
Source: {document["source"]}

Here is the document:
{document["text"]}

Respond with the chunks.
"""

@retry(wait=wait)
def process_document(document):
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    
    vectors = embedder.encode(texts).tolist()

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created in vector_db with {collection.count()} documents")

if __name__ == "__main__":
    documents = preprocess_csv_and_fetch()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")