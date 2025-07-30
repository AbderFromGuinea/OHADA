from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rag_core import EmbeddingModel, process_and_index_pdfs, search_documents, get_answer_with_mistral

# Initialisation FastAPI
app = FastAPI()

# Autoriser l'accès CORS depuis le frontend (Netlify ou autre)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger modèle embeddings
model = EmbeddingModel("sentence-transformers/distiluse-base-multilingual-cased")

# Indexation des PDFs au démarrage
@app.on_event("startup")
def startup_event():
    print("🔄 Indexation des PDFs...")
    try:
        process_and_index_pdfs("./pdfs", model)
        print("✅ Indexation réussie")
    except Exception as e:
        print(f"⚠️ Échec indexation : {e}")

# Endpoint principal : réponse à une question
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question.strip():
        return {"answer": "", "sources": "❌ Question vide."}

    try:
        results = search_documents(question, model, top_k=10)
        if not results:
            return {"answer": "❌ Aucun document trouvé.", "sources": "Aucune source disponible."}

        answer = get_answer_with_mistral(question, results)
        sources = "\n\n---\n\n".join(
            f"📄 **{res['metadata']['source_file']}** (chunk {res['metadata']['chunk_index']}):\n{res['text']}"
            for res in results
        )

        return {"answer": answer, "sources": sources}
    
    except Exception as e:
        return {"answer": f"❌ Erreur : {str(e)}", "sources": "Erreur interne"}
