import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import time
import uuid
from google import genai

st.set_page_config(page_title="ARIL | Stock Advisor", layout="wide", initial_sidebar_state="expanded")

# ------------------- Embedder -------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ------------------- Chroma Setup -------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "chroma_stock_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="stocks")

# ------------------- LOAD DATASETS -------------------
def index_stock_dataset():

    dataset_dir = os.path.join(APP_DIR, "dataset")
    if not os.path.exists(dataset_dir):
        return False, "Dataset folder missing."

    fundamentals = os.path.join(dataset_dir, "stocks_fundamentals.csv")
    sentiment = os.path.join(dataset_dir, "stocks_news_sentiment.csv")
    price = os.path.join(dataset_dir, "stocks_price_history.csv")

    if not (os.path.exists(fundamentals) and os.path.exists(sentiment) and os.path.exists(price)):
        return False, "One or more CSV files missing."

    df_f = pd.read_csv(fundamentals)
    df_s = pd.read_csv(sentiment)
    df_p = pd.read_csv(price)

    all_docs = []

    # Fundamentals
    for _, row in df_f.iterrows():
        text = f"FUNDAMENTALS {row['symbol']} {row['company_name']} PE {row['pe_ratio']} ROE {row['roe']} EPS {row['eps']}"
        all_docs.append((row['symbol'], text, {"symbol": row['symbol'], "type": "fund"}))

    # Sentiment
    for _, row in df_s.iterrows():
        text = f"SENTIMENT {row['symbol']} Headline: {row['headline']} Sentiment: {row['sentiment']} Score {row['score']}"
        all_docs.append((row['symbol'], text, {"symbol": row['symbol'], "type": "sent"}))

    # Price History
    for _, row in df_p.iterrows():
        text = f"PRICE {row['symbol']} Date {row['date']} Close {row['close']} Volume {row['volume']}"
        all_docs.append((row['symbol'], text, {"symbol": row['symbol'], "type": "price"}))

    # Embed & index
    ids = []
    embeddings = []
    docs = []
    metas = []

    for i, (symbol, text, meta) in enumerate(all_docs):
        vector = embedder.encode(text).tolist()
        ids.append(f"id_{i}_{symbol}")
        embeddings.append(vector)
        docs.append(text)
        metas.append(meta)

    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    return True, f"Indexed {len(all_docs)} stock data rows."

# ------------------- Streamlit Layout -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📈 AI Stock Advisor")

st.sidebar.subheader("Dataset Controls")
if st.sidebar.button("Index Stock Dataset"):
    success, msg = index_stock_dataset()
    st.sidebar.write(msg)

API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

# ------------------- Chat -------------------
query = st.chat_input("Ask: Which stock should I buy today?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------- AI Processing -------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":

    user_text = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        st.write("*Analyzing stock data...*")

        # 1. Semantic Search
        q_emb = embedder.encode(user_text).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=8)

        context_docs = []
        for doc in results["documents"][0]:
            context_docs.append(doc)

        # 2. LLM Response
        if API_KEY:
            client_g = genai.Client(api_key=API_KEY)

            prompt = f"""
You are an AI STOCK ANALYST.
Based ONLY on the context dataset below,
suggest the best stock(s) to buy today.

If needed, consider:
- Fundamentals
- Sentiment
- Price momentum

CONTEXT:
{chr(10).join(context_docs)}

QUESTION:
{user_text}

Respond with clear recommendations.
"""

            reply = client_g.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt]
            ).text

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

