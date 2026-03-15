# 📰 NewsLens AI
### A Framework for Analyzing and Summarizing News Articles using Large Language Models

> **Capstone Project** | Kalasalingam Academy of Research and Education  
> Final Review: April 6–10, 2025

---

## 🏗️ System Architecture

```
User Input (URLs)
      ↓
[Scraper] newspaper3k + BeautifulSoup
      ↓
[NLP Chunker] Sliding window chunks (400 words, 80-word overlap)
      ↓
[Embeddings] Sentence Transformers (all-MiniLM-L6-v2)
      ↓
[Vector DB] FAISS IndexFlatIP (cosine similarity)
      ↓
    ┌─────────────────┬─────────────────┐
    │                 │                 │
[Summarizer]    [Q&A / RAG]    [Sentiment + Topics]
 Groq Llama 3   Groq Llama 3    VADER + Groq Llama 3
    │                 │                 │
    └─────────────────┴─────────────────┘
                      ↓
              [Streamlit UI]
```

---

## ⚡ Quick Start (5 minutes)

### Step 1 — Clone and install
```bash
git clone <your-repo-url>
cd newslens
pip install -r requirements.txt
```

### Step 2 — Get your FREE Groq API key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (30 seconds)
3. Click "Create API Key"
4. Copy the key (starts with `gsk_...`)

### Step 3 — Set your API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_your_key_here
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 🎯 Features

| Feature | Technology |
|---------|-----------|
| URL Article Scraping | newspaper3k + BeautifulSoup |
| NLP Text Chunking | Sliding window (400 words, 80 overlap) |
| Semantic Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Storage & Search | FAISS IndexFlatIP |
| Article Summarization | Groq Llama 3 8B |
| Cross-Article Synthesis | Groq Llama 3 8B |
| RAG Q&A Chatbot | FAISS retrieval + Groq Llama 3 |
| Sentiment Analysis | VADER (sentence + article level) |
| Topic Detection | Groq Llama 3 |
| Visualization | Plotly charts |

---

## 📁 Project Structure

```
newslens/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .env                    # Your API keys (don't commit!)
├── README.md               # This file
└── src/
    ├── __init__.py
    ├── scraper.py           # URL scraping (newspaper3k + BS4)
    ├── chunker.py           # NLP text chunking
    ├── vector_store.py      # FAISS vector store + embeddings
    ├── llm_client.py        # Groq API wrapper (Llama 3)
    └── sentiment.py         # VADER sentiment analysis
```

---

## 🛠️ Troubleshooting

### "Could not extract text from URL"
- Some sites block scrapers. Try pasting the article URL directly
- Try a different news source (Reuters, AP News work well)

### "Groq API error"
- Check your API key is correct (starts with `gsk_`)
- Check rate limits at console.groq.com (free tier: 30 req/min)

### Slow first load
- Sentence Transformers downloads the model (~80MB) on first run
- Subsequent runs use the cached model

---

## 📚 References

Based on the research paper:
> *"A Framework for Analyzing and Summarizing News and Articles using Large Language Model"*  
> D. Surendiran Muthukumar et al., ICUIS-2024

Key technologies:
- [Groq](https://groq.com) — Free, fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Sentence Transformers](https://sbert.net) — Semantic embeddings
- [VADER](https://github.com/cjhutto/vaderSentiment) — Sentiment analysis
- [Streamlit](https://streamlit.io) — Web UI framework

---

## 🎓 How It Works (For Your Presentation)

1. **Scraping**: Article text is extracted from URLs using newspaper3k
2. **Chunking**: Text is split into 400-word overlapping chunks (LSA-inspired)
3. **Embeddings**: Each chunk is encoded into a 384-dim vector using MiniLM
4. **FAISS Index**: Vectors stored in a flat inner-product index for O(1) search
5. **RAG**: User queries are embedded → top-5 chunks retrieved → LLM generates answer
6. **Summarization**: Full article text sent to Llama 3 with structured prompt
7. **Sentiment**: VADER compound score computed at article + sentence level
8. **Topics**: LLM extracts entities, themes, and trends from combined text
