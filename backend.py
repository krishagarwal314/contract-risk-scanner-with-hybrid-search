from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import tempfile
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import uvicorn
from sentence_transformers import CrossEncoder


load_dotenv()
qroq_api_key = os.getenv("GROQ_API_KEY")
api_key = os.getenv("pinecone_key")
os.environ["api_key"] = api_key

app = FastAPI()

index_name = 'contract-risk-scanner'

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension = 384,
        metric = "dotproduct",
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

model = init_chat_model("groq:llama-3.1-8b-instant", temperature = 0)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


risk_cats = {
    "payment_terms": {
        "label": "Payment Terms",
        "query": "payment terms invoice due date net 30 net 60 net 90 late payment interest billing schedule",
    },
    "termination": {
        "label": "Termination Clause",
        "query": "termination terminate agreement termination for convenience termination for cause notice period",
    },
    "limitation_of_liability": {
        "label": "Limitation of Liability",
        "query": "limitation of liability liability cap maximum liability damages indirect damages consequential damages",
    },
    "indemnification": {
        "label": "Indemnification",
        "query": "indemnification indemnify hold harmless defend claims third party claims losses liabilities",
    },
    "governing_law": {
        "label": "Governing Law & Jurisdiction",
        "query": "governing law jurisdiction venue dispute resolution arbitration courts applicable law",
    }
}
#these queries are not normal user queries, its a retrieval query. In our model, we'll be relying on syntactic search more than semantic search for better results(therefore keeping the alpha value really low around .1 to .15)


prompt = ChatPromptTemplate.from_template("""
you are a legal contract analyst

analyze the contract excerpts below for the risk category: {category_label}

strict rules:
1. only use information found in the context below
2. do not use any external knowledge.
3. if the clause is not present in the context, mark found as false

context:
{context}

respond ONLY and only with a valid JSON object using these exact keys:
- "found": true or false
- "risk_level": "low", "medium", "high", or "not_found"
- "summary": one sentence describing what the clause says, or "Not found in document"
- "flag": the specific risky detail (e.g. "Net 90 days", "No cap specified", "Auto-renews 12 months") or null

Return only the JSON. No explanation, no markdown, no backticks.


""")


retrievers: dict = {}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete = False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    texts = [doc.page_content for doc in docs]

    bm25 = BM25Encoder()
    bm25.fit(texts)

    namespace = str(uuid.uuid4())

    retriever = PineconeHybridSearchRetriever(
        embeddings = embedding,
        sparse_encoder = bm25,
        index = index,
        top_k = 15,      #updated to 15 after the addition of reranking using a cross encoder in the new commit
        alpha=0.15,
        namespace=namespace,
    )
    retriever.add_texts(texts=texts, namespace=namespace)
    retrievers[namespace] = retriever

    return {"status": "done", "session_id": namespace, "chunks": len(texts)}

class request_format(BaseModel):
    session_id: str

def rerank(query: str, docs: list, top_n = 4) -> list: #new function of reranking in the new commit 
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key = lambda x: x[1], reverse = True)
    return [doc for doc, score in scored_docs[:top_n]]

@app.post("/scan")
async def scan(request: request_format):
    retriever = retrievers.get(request.session_id)

    if retriever is None:
        return{"error": "error"}
    
    chain = prompt | model | JsonOutputParser()

    results = {}

    for key, data in risk_cats.items():
        docs = retriever.invoke(data["query"])    #we get top 15 chunks based on the dot product 
        reranked_docs = rerank(query = data["query"], docs = docs, top_n = 4)  #top 4 docs based on the results of cross encoder
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
        try:
            result = chain.invoke({
                "category_label": data["label"],
                "context": context,
            })
        except Exception as e:
            print(f"ERROR on {key}: {e}")  
            result = {
                "found": False,
                "risk_level": "not_found",
                "summary": "extraction failed",
                "flag": None,
            }
        results[key] = {"label": data["label"], **result}

    
    high = sum(1 for v in results.values() if v.get("risk_level") == "high")
    medium = sum(1 for v in results.values() if v.get("risk_level") == "medium")
    low = sum(1 for v in results.values() if v.get("risk_level") == "low")


    return {
        "summary": {"high": high, "medium": medium, "low": low},
        "risks": results
    }

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
