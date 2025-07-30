import os
import hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pymongo import MongoClient
from loguru import logger
import pinecone
# import fitz  # PyMuPDF
import numpy as np
import pytesseract
import cv2
from sentence_transformers import SentenceTransformer
import re

# Charger les variables d'environnement
load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,  # change to your vector size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",       # or "gcp"
            region="us-east-1" # choose your region
        ),
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Initialisation MongoDB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError("Variable d'environnement MONGO_URI manquante")
client = MongoClient(MONGO_URI)
db = client["rag_database"]
chunks_collection = db["chunks"]
'''
# --- Classes et fonctions utilitaires ---

class PDFProcessor:
    def __init__(self, use_ocr=True):
        self.use_ocr = use_ocr
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except Exception:
            self.ocr_available = False
            logger.warning("Tesseract non disponible, OCR désactivé.")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        logger.info(f"Extraction du texte du PDF: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if len(text.strip()) < 50 and self.ocr_available:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6 -l fra")
            full_text += f"\n\n--- Page {page_num+1} ---\n{text.strip()}"
        doc.close()
        return full_text.strip()

def chunk_text(text: str, chunk_size=1200, chunk_overlap=300) -> List[str]:
    """Chunking strategy de base - conservée pour compatibilité"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            chunks.append(current_chunk)
            # overlap
            current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para if chunk_overlap > 0 else para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def enhanced_legal_chunking(text, filename, max_chunk_size=1000, overlap=200):
    """
    Enhanced chunking specifically designed for legal documents with proper None handling.
    """
    if not text or text.strip() == "":
        return []
    
    chunks = []
    
    # Define legal document patterns
    patterns = {
        'article': r'\b(?:Article|ARTICLE)\s+\d+',
        'section': r'\b(?:Section|SECTION|§)\s*\d+',
        'chapter': r'\b(?:Chapter|CHAPTER)\s+\d+',
        'clause': r'\b(?:Clause|CLAUSE)\s+\d+',
        'paragraph': r'\b(?:Paragraph|PARAGRAPH|Para\.?)\s*\d+',
        'subsection': r'\b(?:Subsection|SUBSECTION)\s*\d+',
        'part': r'\b(?:Part|PART)\s+[IVX]+|\b(?:Part|PART)\s+\d+',
        'title': r'\b(?:Title|TITLE)\s+\d+',
        'numbered_item': r'^\s*\d+\.\s+',
        'lettered_item': r'^\s*[a-z]\)\s+',
        'roman_numeral': r'^\s*[ivx]+\)\s+',
    }
    
    # Try to split by major legal structures first
    major_patterns = ['article', 'chapter', 'title', 'part']
    
    for pattern_name in major_patterns:
        pattern = patterns[pattern_name]
        parts = re.split(f'({pattern})', text, flags=re.IGNORECASE | re.MULTILINE)
        
        if len(parts) > 1:  # Found splits
            current_chunk = ""
            
            for i, part in enumerate(parts):
                # Skip None values and empty strings
                if part is None:
                    continue
                    
                # Strip whitespace and skip empty parts
                part = part.strip()
                if not part:
                    continue
                
                # If this part matches our pattern, it's a header
                if re.match(pattern, part, re.IGNORECASE):
                    # Save previous chunk if it exists and is substantial
                    if current_chunk.strip() and len(current_chunk.strip()) > 50:
                        chunks.extend(_split_large_chunk(current_chunk.strip(), max_chunk_size, overlap))
                    
                    # Start new chunk with this header
                    current_chunk = part + "\n"
                else:
                    # Add content to current chunk
                    current_chunk += part + "\n"
                    
                    # If chunk is getting too large, split it
                    if len(current_chunk) > max_chunk_size * 1.5:
                        # Find a good breaking point
                        break_point = _find_break_point(current_chunk, max_chunk_size)
                        if break_point > 0:
                            chunks.append(current_chunk[:break_point].strip())
                            current_chunk = current_chunk[break_point-overlap:].strip() + "\n"
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.extend(_split_large_chunk(current_chunk.strip(), max_chunk_size, overlap))
            
            # If we found good chunks, return them
            if chunks and len(chunks) > 1:
                return [chunk for chunk in chunks if chunk.strip()]
    
    # Fallback: split by smaller structures
    smaller_patterns = ['section', 'subsection', 'clause', 'paragraph']
    
    for pattern_name in smaller_patterns:
        pattern = patterns[pattern_name]
        parts = re.split(f'({pattern})', text, flags=re.IGNORECASE | re.MULTILINE)
        
        if len(parts) > 1:
            current_chunk = ""
            
            for part in parts:
                # Skip None values and empty strings
                if part is None:
                    continue
                    
                # Strip whitespace and skip empty parts
                part = part.strip()
                if not part:
                    continue
                
                if len(current_chunk + part) > max_chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part + "\n"
                else:
                    current_chunk += part + "\n"
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            if chunks and len(chunks) > 1:
                return [chunk for chunk in chunks if chunk.strip()]
    
    # Final fallback: split by paragraphs and numbered items
    paragraphs = re.split(r'\n\s*\n|\n\s*\d+\.\s+|\n\s*[a-z]\)\s+', text)
    
    # Filter out None values and empty strings
    paragraphs = [p.strip() for p in paragraphs if p is not None and p.strip()]
    
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        if paragraph is None:
            continue
            
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if len(current_chunk + paragraph) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If still no good chunks, split by sentences
    if not chunks or len(chunks) == 1:
        return _split_large_chunk(text, max_chunk_size, overlap)
    
    return [chunk for chunk in chunks if chunk.strip()]


def _find_break_point(text, max_size):
    """Find a good place to break text, preferring sentence or paragraph boundaries."""
    if len(text) <= max_size:
        return len(text)
    
    # Look for paragraph break
    para_break = text.rfind('\n\n', 0, max_size)
    if para_break > max_size * 0.7:
        return para_break
    
    # Look for sentence break
    sent_break = max(
        text.rfind('. ', 0, max_size),
        text.rfind('.\n', 0, max_size),
        text.rfind('! ', 0, max_size),
        text.rfind('? ', 0, max_size)
    )
    if sent_break > max_size * 0.7:
        return sent_break + 1
    
    # Look for clause break
    clause_break = max(
        text.rfind(', ', 0, max_size),
        text.rfind('; ', 0, max_size),
        text.rfind(': ', 0, max_size)
    )
    if clause_break > max_size * 0.8:
        return clause_break + 1
    
    # Last resort: break at word boundary
    word_break = text.rfind(' ', 0, max_size)
    if word_break > 0:
        return word_break
    
    return max_size


def _split_large_chunk(text, max_chunk_size, overlap):
    """Split a large chunk into smaller pieces with overlap."""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Find a good breaking point
        break_point = _find_break_point(text[start:end], max_chunk_size)
        chunk_end = start + break_point
        
        chunk = text[start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + break_point - overlap, start + 1)
    
    return [chunk for chunk in chunks if chunk.strip()]
'''
def clean_metadata_for_pinecone(metadata):
    """
    Clean metadata to ensure compatibility with Pinecone requirements.
    Pinecone accepts: strings, numbers, booleans, or lists of strings.
    Removes None values and converts unsupported types.
    """
    cleaned = {}
    
    for key, value in metadata.items():
        if value is None:
            # Skip null values entirely
            continue
        elif isinstance(value, (str, int, float, bool)):
            # Direct assignment for supported types
            cleaned[key] = value
        elif isinstance(value, list):
            # Filter out None values from lists and ensure all items are strings
            cleaned_list = [str(item) for item in value if item is not None]
            if cleaned_list:  # Only add if list is not empty
                cleaned[key] = cleaned_list
        else:
            # Convert other types to string
            cleaned[key] = str(value)
    
    return cleaned

def extract_chunk_metadata(chunk: str, source_file: str) -> dict:
    """
    Extrait des métadonnées enrichies d'un chunk juridique.
    Returns metadata with no None values to avoid Pinecone errors.
    """
    metadata = {
        "has_article": False,
        "has_chapter": False,
        "document_type": "legal"
    }
    
    # Détecter les articles - patterns simplifiés
    article_match = re.search(r'Article\s+(\d+(?:\s*[a-zA-Z])?)', chunk, re.IGNORECASE)
    if not article_match:
        article_match = re.search(r'Art\.?\s*(\d+(?:\s*[a-zA-Z])?)', chunk, re.IGNORECASE)
    
    if article_match:
        metadata["has_article"] = True
        metadata["article_number"] = article_match.group(1)
    
    # Détecter les chapitres
    chapter_match = re.search(r'(Chapitre\s+\d+[^\n]*)', chunk, re.IGNORECASE)
    if not chapter_match:
        chapter_match = re.search(r'(CHAPITRE\s+\d+[^\n]*)', chunk, re.IGNORECASE)
    
    if chapter_match:
        metadata["has_chapter"] = True
        metadata["chapter_info"] = chapter_match.group(1)
    
    # Déterminer le type de document basé sur le nom du fichier
    source_lower = source_file.lower()
    if "traite" in source_lower:
        metadata["document_type"] = "traite"
    elif "mediation" in source_lower:
        metadata["document_type"] = "mediation"
    elif "aum" in source_lower:
        metadata["document_type"] = "acte_uniforme"
    
    return metadata

def file_hash_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

class EmbeddingModel:
    def __init__(self, model_name="distiluse-base-multilingual-cased"):
        logger.info(f"Chargement du modèle embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=False)

def is_pdf_already_indexed(file_hash: str) -> bool:
    # Recherche dans MongoDB si hash existe
    return chunks_collection.find_one({"file_hash": file_hash}) is not None

# --- Fonction principale ---

def process_and_index_pdfs(pdf_folder_path: str, embedding_model: EmbeddingModel):
    """
    Extrait le texte des PDFs, le segmente avec la stratégie améliorée pour documents juridiques,
    génère les embeddings et les indexe dans MongoDB + Pinecone.
    """
    pdf_folder = Path(pdf_folder_path)
    if not pdf_folder.exists():
        logger.error(f"Dossier PDF non trouvé: {pdf_folder_path}")
        return

    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning("Aucun fichier PDF trouvé dans le dossier.")
        return

    pdf_processor = PDFProcessor(use_ocr=True)
    total_chunks_indexed = 0

    for pdf_file in pdf_files:
        h = file_hash_md5(pdf_file)
        if is_pdf_already_indexed(h):
            logger.info(f"PDF déjà indexé (hash: {h}): {pdf_file.name} - saut du retraitement")
            continue

        logger.info(f"🔄 Traitement de {pdf_file.name} avec chunking juridique amélioré...")
        
        # Extraction du texte
        text = pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Utiliser la stratégie de chunking améliorée pour les documents juridiques
        chunks = enhanced_legal_chunking(text, pdf_file.name)
        
        if not chunks:
            logger.warning(f"Aucun chunk généré pour {pdf_file.name}")
            continue

        # Log des premiers chunks pour débogage
        logger.info(f"Premier chunk exemple pour {pdf_file.name}: {chunks[0][:200]}...")

        # Génération des embeddings
        embeddings = embedding_model.encode(chunks)

        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = f"{pdf_file.stem}_chunk{i}"
            
            # Extraire des métadonnées enrichies du chunk
            chunk_metadata = extract_chunk_metadata(chunk, pdf_file.name)
            
            # Métadonnées pour MongoDB (peuvent contenir des valeurs None)
            mongo_metadata = {
                "source_file": pdf_file.name,
                "chunk_index": i,
                "file_hash": h,
                "chunk_length": len(chunk),
                **chunk_metadata
            }
            
            # Métadonnées nettoyées pour Pinecone (sans valeurs None)
            pinecone_metadata = clean_metadata_for_pinecone(mongo_metadata)
            
            # Convertir l'embedding en liste si nécessaire
            embedding_vector = embeddings[i]
            if not isinstance(embedding_vector, list):
                embedding_vector = embedding_vector.tolist()
            
            vectors.append({
                "id": vector_id,
                "values": embedding_vector,
                "metadata": pinecone_metadata
            })
            
            # Sauvegarder dans MongoDB avec métadonnées complètes (incluant les None)
            chunks_collection.update_one(
                {"_id": vector_id},
                {"$set": {"text": chunk, **mongo_metadata}},
                upsert=True
            )

        # Indexer dans Pinecone par batches pour éviter les timeouts
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch)
                logger.info(f"Batch {i//batch_size + 1} indexé avec succès ({len(batch)} vecteurs)")
            except Exception as e:
                logger.error(f"Erreur lors de l'indexation du batch {i//batch_size + 1}: {e}")
                # Essayer un par un en cas d'erreur de batch
                for vector in batch:
                    try:
                        index.upsert(vectors=[vector])
                    except Exception as ve:
                        logger.error(f"Erreur sur le vecteur {vector['id']}: {ve}")
        
        total_chunks_indexed += len(chunks)
        logger.info(f"✅ Indexé {len(chunks)} chunks pour {pdf_file.name}")

    logger.info(f"🎉 Indexation terminée, total chunks insérés: {total_chunks_indexed}")


import re
from sentence_transformers import CrossEncoder

# Optionnel : charge le reranker si dispo
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker = True
except:
    reranker = None
    use_reranker = False
    print("[⚠️] Reranker non chargé. Recherche sémantique uniquement.")

# --- Fonction pour extraire un numéro d’article ---
def extract_article_number(question):
    match = re.search(r"(article\s+)?(\d{1,4})([a-zA-Z]*)", question, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None

# --- Fonction principale avec reranking ---
def search_documents(query: str, embedding_model, top_k=10):
    # Étape 1 : Recherche directe par numéro d’article
    article_num = extract_article_number(query)
    if article_num:
        print(f"[🔎] Recherche directe pour l'article {article_num}")
        article_matches = list(chunks_collection.find({
            "$or": [
                {"article_number": {"$regex": f"^{article_num}\\b", "$options": "i"}},
                {"text": {"$regex": f"Article\\s+{article_num}\\b", "$options": "i"}}
            ]
        }).limit(top_k))

        if article_matches:
            return [{
                "id": str(m["_id"]),
                "score": 1.0,
                "metadata": m,
                "text": m["text"]
            } for m in article_matches]

    # Étape 2 : Recherche sémantique
    print("[🧠] Recherche sémantique...")
    query_embedding = embedding_model.encode([query])[0]
    if not isinstance(query_embedding, list):
        query_embedding = query_embedding.tolist()

    response = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    raw_results = []
    for match in response.matches:
        chunk_data = chunks_collection.find_one({"_id": match.id})
        if not chunk_data or "text" not in chunk_data:
            continue
        raw_results.append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
            "text": chunk_data["text"]
        })

    # Étape 3 : Re-ranking (si activé)
    if use_reranker and len(raw_results) > 0:
        print("[🔁] Re-ranking des résultats...")
        pairs = [(query, chunk["text"]) for chunk in raw_results]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)
        return [x[1] for x in reranked[:top_k]]

    # Fallback : retourne les résultats initiaux si pas de reranker
    return raw_results



# ---fonction de chat history---

from collections import deque

class ChatHistory:
    def __init__(self, max_turns=5):
        self.history = deque(maxlen=max_turns * 2)  # user + assistant messages

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get(self):
        return list(self.history)

    def reset(self):
        self.history.clear()



import os
import requests

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

def get_natural_answer_with_gemini(question, context_chunks, temperature=0.7):
    """
    Envoie une question + contexte à l'API Gemini et retourne une réponse naturelle.
    """
    if GEMINI_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY non défini dans les variables d'environnement.")
    
    context_text = "\n\n".join([f"- {chunk['text']}" for chunk in context_chunks])

    prompt = f"""
Tu es un assistant juridique pédagogue et clair. En te basant uniquement sur le contexte suivant (textes OHADA), réponds à la question en français simple et professionnel. Sois structuré, calme, et facile à comprendre. Ne fais pas de supposition.Suggère les lieux ou entités à consulter pour plus d'informations.

Contexte :
{context_text}

Question :
{question}
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 512
        }
    }

    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    

    if response.status_code == 200:
        reply = response.json()
        try:
            return reply["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "❌ La réponse de Gemini est vide ou mal formatée."
    else:
        return f"❌ Erreur Gemini ({response.status_code}) : {response.text}"


# --- à ajouter en haut ---
import os
from mistralai import Mistral  # client officiel Mistral
from dotenv import load_dotenv
from typing import List

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def get_answer_with_mistral(
    question: str,
    context_chunks: List[dict],
    temperature: float = 0.7,
    chat_history=None
):
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY non défini dans .env")

    # Contexte RAG
    context_text = "\n\n".join(f"- {chunk['text']}" for chunk in context_chunks)

    system_prompt = (
        "Tu es un assistant juridique spécialisé en droit OHADA.\n"
        "Ta réponse doit être claire, pédagogique et uniquement fondée sur le contexte fourni ci-dessous. "
        "Si le contexte ne contient pas la réponse, indique-le.\n"
        "Lorsque tu cites une règle ou un article, mentionne son numéro et le nom du texte (ex : Traité OHADA, Article 42).\n\n"
        "Si possible suggère les lieux ou entités à contacter pour plus d'informations.\n"
        "Lorsque la question concerne la création d'entreprise, merci d'inviter l'utilisateur à visiter le site de l'APIP Guinée.\n"
        "Si tu es sûr à 95% d'avoir la réponse dans tes connaissances personnelles, n'hésite pas à répondre à la question."
    )

    # Messages à envoyer au modèle
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history.get())

    messages.append({
        "role": "user",
        "content": f"Contexte :\n{context_text}\n\nQuestion : {question}"
    })

    # Appel Mistral
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
            stream=False
        )

    # Extraire contenu
    try:
        content = resp.choices[0].message.content
    except Exception as e:
        content = None

    # Ajouter à l’historique
    if chat_history and content:
        chat_history.add_message("user", question)
        chat_history.add_message("assistant", content)

    return content or "❌ Réponse non disponible de Mistral."
