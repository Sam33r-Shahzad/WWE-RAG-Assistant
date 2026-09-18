import os
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer
import os
from litellm import completion

load_dotenv(override=True)



MODEL = "openrouter/deepseek/deepseek-chat"

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

collection_name = "docs"
embedding_model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)

wait = wait_exponential(multiplier=1, min=10, max=240)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 10
FINAL_K = 5
SYSTEM_PROMPT = """
You are WWEish, a knowledgeable and professional assistant representing a comprehensive WWE History and Match Knowledge Base. You have the data of complete year 2023 until now.
You are chatting with a user about professional wrestling, matches, wrestlers, promotions, and events.
Answer strictly based on the provided context in a clean, professional paragraph format without using bullet points or numbered lists. 
Rely entirely on the retrieved chunks for match outcomes and event details. If a specific date or detail is mentioned in the chunk, state it accurately as provided in the context without adding external assumptions.
Never hallucinate or invent information. If the answer is not present in the context, state simply: "I don't know about this."
Do not discuss API keys, technical configurations, or system limitations.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question accurately, relevantly, and completely in a single cohesive paragraph.
"""

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = completion(model=MODEL, messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
    
        valid_order = [i for i in order if 1 <= i <= len(chunks)]
        reranked_chunks = [chunks[i - 1] for i in valid_order]
        missing_chunks = [c for c in chunks if c not in reranked_chunks]
        return reranked_chunks + missing_chunks
    except Exception:
        return chunks

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata.get('source', 'knowledge base')}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


# @retry(wait=wait)
def rewrite_query(question, history=[]):
    message = f"""
You are in a conversation with a user, answering questions about WWE history, matches, wrestlers, and events.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    # Free local embedding using sentence-transformers
    query = embedder.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]



# @retry(wait=wait)
# def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
#     chunks = fetch_context(question)
#     messages = make_rag_messages(question, history, chunks)
#     response = completion(model=MODEL, messages=messages)
#     return response.choices[0].message.content, chunks



def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    print(f"\n[USER QUESTION]: {question}")
    # 1. Query Rewrite step
    rewritten_question = rewrite_query(question)
    print(f"[REWRITTEN QUERY]: {rewritten_question}")
    
    chunks1 = fetch_context_unranked(question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    # 2. Rerank step
    reranked = rerank(question, chunks)
    final_chunks = reranked[:FINAL_K]
    
    print(f"[RETRIEVED & RERANKED CHUNKS COUNT]: {len(final_chunks)}")
    for i, c in enumerate(final_chunks):
        print(f"   -> Top Chunk {i+1} preview: {c.page_content[:100]}...")

    messages = make_rag_messages(question, history, final_chunks)
    response = completion(model=MODEL, messages=messages)
    answer_text = response.choices[0].message.content
    print(f"[FINAL ANSWER]: {answer_text}\n")
    
    return answer_text, final_chunks