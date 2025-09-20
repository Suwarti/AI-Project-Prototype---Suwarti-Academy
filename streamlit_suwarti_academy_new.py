# streamlit_suwarti_academy.py
# ------------------------------------------------------------
# RAG Chat (Gemini + ChromaDB) dari folder PDF lokal (Suwarti Academy).
# Versi aman untuk Streamlit Cloud:
# - Gunakan pysqlite3-binary (patch sqlite3)
# - Fallback ke Chroma in-memory jika PersistentClient gagal
# - Tidak menampilkan daftar PDF, context viewer, atau pesan indexing selesai
# ------------------------------------------------------------

# ---- SQLite patch for Streamlit Cloud (MUST be first) ----
import sys
try:
    import pysqlite3  # from pysqlite3-binary
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# -----------------------------------------------------------

import os
import glob
import re
import unicodedata
from io import BytesIO
from typing import List, Tuple

# Paksa UTF-8 agar emoji / non-ASCII aman
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

# (opsional) muat .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Google Gemini SDK (google-genai)
try:
    from google import genai
    GENAI_OK = True
except Exception:
    GENAI_OK = False

# ---------------- UI & Style ----------------
st.set_page_config(page_title="Suwarti Academy", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .stChatMessage { max-width: 900px; margin-left:auto; margin-right:auto; }
    .block-container { padding-top: 1rem; }
    .bubble-user { background: #0b6efd10; border: 1px solid #0b6efd30; padding: .75rem 1rem; border-radius: 12px; color: #0b6efd; }
    .bubble-bot  { background: #f6f6f6;   border: 1px solid #e6e6e6;   padding: .75rem 1rem; border-radius: 12px; color: #111; }
    code, pre { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Project Prototype - Suwarti Academy")
st.caption("LLM Based Tools and Gemini API Integration")

# ---------------- Helpers ----------------
def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or ""
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

def ensure_utf8(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

def extract_text_from_pdf(file_obj: BytesIO) -> str:
    reader = PdfReader(file_obj)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return ensure_utf8("\n".join(text))

def extract_text_from_pdf_path(path: str) -> str:
    with open(path, "rb") as f:
        return extract_text_from_pdf(f)

# ---------- CLEANSING ----------
def cleanse_text(text: str) -> str:
    """
    Cleansing ringan untuk RAG:
    - Normalisasi unicode (NFKC), normalisasi newline
    - Hilangkan control chars
    - Gabungkan kata terpotong 'kata-\nlanjut' -> 'katalanjut'
    - Rapikan spasi; kompres baris kosong
    """
    if not text:
        return ""
    text = ensure_utf8(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    cleaned, last_blank = [], False
    for ln in lines:
        if ln == "":
            if not last_blank:
                cleaned.append("")
            last_blank = True
        else:
            if not cleaned or ln != cleaned[-1]:
                cleaned.append(ln)
            last_blank = False
    return "\n".join(cleaned).strip()

def chunk_texts(text: str, chunk_size=1800, chunk_overlap=250) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True,
    )
    return [d.page_content for d in splitter.create_documents([text])]

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        if not GENAI_OK:
            raise RuntimeError("google-genai belum terpasang. Jalankan: pip install google-genai")
        self.client = genai.Client(api_key=api_key)
        self.model = model
    def __call__(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [ensure_utf8(t) for t in texts]
        resp = self.client.models.embed_content(model=self.model, contents=texts)
        return [emb.values for emb in resp.embeddings]

@st.cache_resource(show_spinner=False)
def chroma_client(path: str):
    try:
        return chromadb.PersistentClient(path=path)
    except Exception:
        return chromadb.Client()

def get_or_create_collection(client, name: str, embedder):
    return client.get_or_create_collection(name=name, embedding_function=embedder)

def add_docs(col, docs: List[str], source: str):
    if not docs:
        return
    docs = [ensure_utf8(d) for d in docs]
    source = ensure_utf8(source)
    start = col.count()
    col.add(
        ids=[str(start + i) for i in range(len(docs))],
        documents=docs,
        metadatas=[{"source": source, "chunk": i} for i in range(len(docs))]
    )

def retrieve(col, q: str, k: int = 10):
    q = ensure_utf8(q)
    res = col.query(query_texts=[q], n_results=k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))

def filter_contexts(contexts: List[Tuple[str, dict, float]], max_distance: float):
    filtered = []
    for doc, meta, dist in contexts:
        try:
            d = float(dist)
        except Exception:
            d = 999.0
        if doc and str(doc).strip() and d <= max_distance:
            filtered.append((doc, meta, d))
    return filtered

def order_contexts(contexts: List[Tuple[str, dict, float]]) -> List[Tuple[str, dict, float]]:
    try:
        return sorted(contexts, key=lambda x: (x[1].get("source", ""), int(x[1].get("chunk", 0))))
    except Exception:
        return contexts

# ---------- PROMPTS ----------
def build_prompt_pdf_first(user_q: str, contexts: List[Tuple[str, dict, float]]) -> str:
    contexts = order_contexts(contexts)
    ctx = ""
    for doc, meta, dist in contexts:
        ctx += f"{doc}\n\n"
    return (
        "Role: You are a helpful assistant.\n"
        "Instructions:\n"
        "- Detect the question’s language and answer in the same language.\n"
        "- Use ALL of the CONTEXT chunks below together as ONE continuous reference.\n"
        "- Preserve and reproduce the original structure: section titles, numbering, bullet points, and paragraph explanations.\n"
        "- Provide the most detailed, step-by-step answer possible. Do NOT shorten, compress, or omit sections.\n"
        "- If a chunk is continuation of a previous section, merge them naturally.\n"
        "- Do NOT mention 'context' or 'source'.\n\n"
        f"=== CONTEXT ===\n{ctx}\n"
        f"=== QUESTION ===\n{user_q}\n\n"
        "Write the most DETAILED and COMPLETE answer."
    )

def build_prompt_fallback(user_q: str) -> str:
    return (
        "Role: You are a helpful assistant.\n"
        "Instructions:\n"
        "- Detect the question’s language and answer in the same language.\n"
        "- Provide a clear and informative answer using accurate general knowledge.\n\n"
        f"Question: {user_q}\n\n"
        "Write your answer now."
    )

def ask_gemini(api_key: str, prompt: str, model: str = "gemini-1.5-flash") -> str:
    if not GENAI_OK:
        return "Paket `google-genai` belum terpasang. Jalankan: pip install google-genai"
    client = genai.Client(api_key=api_key)
    try:
        prompt = ensure_utf8(prompt)
        r = client.models.generate_content(model=model, contents=prompt)
        return getattr(r, "text", "") or "(model tidak mengembalikan teks)"
    except Exception as e:
        return f"Gagal memanggil model: {e}"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    pdf_folder = st.text_input("📂 Folder PDF lokal", value="./suwarti_pdfs")
    persist_path = st.text_input("💾 Folder Chroma (abaikan jika in-memory)", value="suwarti_chroma")
    collection_name = st.text_input("🏷️ Nama koleksi", value="suwarti_kb_pdf")

    st.markdown("---")
    k_results = st.slider("Top-K retrieval", 3, 12, 10)
    relevance_thr = st.slider("Relevance threshold (cosine distance)", 0.0, 1.0, 0.70, step=0.05)
    chunk_size = st.number_input("Chunk size", 500, 4000, 1800, step=50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 250, step=10)

    st.markdown("---")
    model_name = st.selectbox("Model generasi", ["gemini-1.5-flash", "gemini-1.5-pro"])
    api_key = st.text_input("🔑 GEMINI API Key", type="password", value=(get_api_key() or ""))

    rebuild = st.button("🚀 Index ulang dari folder")

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------- Indexing ----------------
def list_local_pdfs(folder: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)])

if rebuild:
    if not api_key:
        st.error("Masukkan GEMINI API Key terlebih dahulu.")
    elif not GENAI_OK:
        st.error("`pip install google-genai` belum dilakukan.")
    else:
        try:
            os.makedirs(pdf_folder, exist_ok=True)
            pdf_paths = list_local_pdfs(pdf_folder)
            embedder = GeminiEmbeddingFunction(api_key=api_key)
            client = chroma_client(persist_path)
            col = get_or_create_collection(client, collection_name, embedder)

            for path in pdf_paths:
                try:
                    raw = extract_text_from_pdf_path(path) or ""
                    text = cleanse_text(raw)
                    if not text.strip():
                        continue
                    chunks = chunk_texts(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    add_docs(col, chunks, source=os.path.basename(path))
                except Exception as e:
                    st.error(ensure_utf8(f"Gagal memproses {os.path.basename(path)}: {e}"))
        except Exception as e:
            st.error(ensure_utf8(f"Gagal index dari folder: {e}"))

# ---------------- Chat UI ----------------
st.subheader("Ask Suwarti Academy")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        css = "bubble-user" if m["role"] == "user" else "bubble-bot"
        st.markdown(f"<div class='{css}'>{ensure_utf8(m['content'])}</div>", unsafe_allow_html=True)

user_q = st.chat_input("Tulis pertanyaan kamu…")
if user_q:
    user_q = ensure_utf8(user_q)
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(f"<div class='bubble-user'>{user_q}</div>", unsafe_allow_html=True)

    if not api_key:
        answer = "Masukkan GEMINI API Key di sidebar."
    elif not GENAI_OK:
        answer = "Paket `google-genai` belum terpasang. Jalankan: pip install google-genai"
    else:
        try:
            client = chroma_client(persist_path)
            try:
                tmp_col = client.get_collection(collection_name)
                count = tmp_col.count()
            except Exception:
                count = 0

            if count == 0:
                prompt = build_prompt_fallback(user_q)
                answer = ask_gemini(api_key, prompt, model=model_name)
            else:
                embedder = GeminiEmbeddingFunction(api_key=api_key)
                col = get_or_create_collection(client, collection_name, embedder)
                contexts_raw = retrieve(col, user_q, k=k_results)
                contexts = filter_contexts(contexts_raw, relevance_thr)
                contexts = order_contexts(contexts)

                if not contexts:
                    prompt = build_prompt_fallback(user_q)
                    answer = ask_gemini(api_key, prompt, model=model_name)
                else:
                    prompt = build_prompt_pdf_first(user_q, contexts)
                    answer = ask_gemini(api_key, prompt, model=model_name)
        except Exception as e:
            answer = f"Terjadi error: {ensure_utf8(str(e))}"

    with st.chat_message("assistant"):
        st.markdown(f"<div class='bubble-bot'>{ensure_utf8(answer)}</div>", unsafe_allow_html=True)

    st.session_state["messages"].append({"role": "assistant", "content": ensure_utf8(answer)})
