"""
app.py - NewsLens AI
Primary Flow: User enters topic → app fetches live news → summarizes + analyzes
Secondary Flow: Manual URL input (optional, advanced)
"""

import streamlit as st
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NewsLens AI", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main-header{font-size:2.6rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#8b5cf6,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center;padding:.5rem 0 .1rem;letter-spacing:-.5px}
.sub-header{text-align:center;color:#6b7280;font-size:.92rem;margin-bottom:.5rem}
.metric-card{background:linear-gradient(135deg,#f8fafc,#f1f5f9);border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem;text-align:center}
.metric-value{font-size:1.8rem;font-weight:700;color:#4f46e5}
.metric-label{font-size:.78rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
.article-card{background:#fff;border:1px solid #e5e7eb;border-left:4px solid #6366f1;border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.article-card h4{margin:0 0 .25rem;color:#1e293b;font-size:.92rem;font-weight:600}
.article-card p{margin:0;color:#6b7280;font-size:.78rem}
.summary-box{background:#fafafa;border:1px solid #e5e7eb;border-radius:10px;padding:1.2rem;line-height:1.75;font-size:.91rem;color:#374151}
.synthesis-box{background:linear-gradient(135deg,#f5f3ff,#ede9fe);border:1px solid #c4b5fd;border-radius:12px;padding:1.3rem;line-height:1.75;font-size:.93rem;color:#3b0764}
.chat-user{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;padding:.75rem 1.1rem;border-radius:18px 18px 4px 18px;margin:.5rem 0 .5rem auto;max-width:78%;font-size:.88rem;display:block}
.chat-assistant{background:#f1f5f9;color:#1e293b;padding:.75rem 1.1rem;border-radius:18px 18px 18px 4px;margin:.5rem auto .5rem 0;max-width:85%;font-size:.88rem;line-height:1.65;display:block}
.source-badge{display:inline-block;background:#f0fdf4;color:#166534;border:1px solid #bbf7d0;border-radius:6px;padding:.15rem .5rem;font-size:.72rem;font-weight:600;margin-right:.4rem}
.info-banner{background:linear-gradient(135deg,#ede9fe,#ddd6fe);border:1px solid #c4b5fd;border-radius:10px;padding:.7rem 1.1rem;font-size:.83rem;color:#5b21b6;margin-bottom:1rem}
.sentiment-positive{color:#16a34a;font-weight:600}
.sentiment-negative{color:#dc2626;font-weight:600}
.sentiment-neutral{color:#d97706;font-weight:600}
.stButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;border-radius:8px;padding:.45rem 1.4rem;font-weight:600}
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "articles": [], "summaries": [], "synthesis": "", "topics_text": "",
        "vector_store": None, "chat_history": [], "processed": False,
        "groq_client": None, "current_topic": "", "newsapi_key": "", "gnews_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


@st.cache_resource(show_spinner=False)
def get_vector_store():
    from src.vector_store import VectorStore
    return VectorStore()


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("#### 🤖 AI Summarization")
    api_key = st.text_input("Groq API Key (required)", type="password",
        value=os.getenv("GROQ_API_KEY", ""), placeholder="gsk_...",
        help="Free at console.groq.com — powers all AI summaries & Q&A")
    if api_key:
        from groq import Groq
        st.session_state.groq_client = Groq(api_key=api_key)
        st.success("✅ Groq ready!")
    else:
        st.markdown('<div class="info-banner">👆 Required: <a href="https://console.groq.com" target="_blank">console.groq.com</a></div>', unsafe_allow_html=True)

    st.markdown("#### 📰 News Fetching (Optional Boost)")
    st.markdown('<div class="info-banner" style="font-size:0.78rem">Adding these keys gives <strong>more accurate</strong> and <strong>more articles</strong>. Both are free.</div>', unsafe_allow_html=True)

    newsapi_key = st.text_input("NewsAPI Key (optional)",
        type="password", value=os.getenv("NEWSAPI_KEY", ""),
        placeholder="abc123...",
        help="Free at newsapi.org — 100 req/day, very accurate topic search")
    if newsapi_key:
        st.session_state.newsapi_key = newsapi_key
        st.success("✅ NewsAPI ready!")

    gnews_key = st.text_input("GNews API Key (optional)",
        type="password", value=os.getenv("GNEWS_KEY", ""),
        placeholder="abc123...",
        help="Free at gnews.io — 100 req/day, extra article source")
    if gnews_key:
        st.session_state.gnews_key = gnews_key
        st.success("✅ GNews ready!")

    with st.expander("🔑 Where to get free API keys"):
        st.markdown("""
**Groq** (required) — AI summaries
→ [console.groq.com](https://console.groq.com)

**NewsAPI** (optional boost) — accurate news search
→ [newsapi.org/register](https://newsapi.org/register)

**GNews** (optional boost) — extra news source
→ [gnews.io](https://gnews.io)

All three are **completely free**.
        """)

    if st.session_state.processed:
        st.markdown("---")
        st.markdown("### 📊 Current Session")
        st.markdown(f"🔍 **Topic:** {st.session_state.current_topic}")
        st.markdown(f"📄 **{len(st.session_state.articles)}** articles loaded")
        st.markdown(f"💬 **{len(st.session_state.chat_history)//2}** questions asked")
        if st.button("🗑️ New Search", use_container_width=True):
            for k in ["articles","summaries","synthesis","topics_text","chat_history","processed","current_topic"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else ("" if isinstance(st.session_state[k], str) else False)
            st.session_state.vector_store = None
            st.rerun()

    st.markdown("---")
    st.markdown('<small style="color:#9ca3af">NewsLens AI • Capstone Project<br>Kalasalingam Academy of Research and Education</small>', unsafe_allow_html=True)


# Header
st.markdown('<div class="main-header">📰 NewsLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Framework for Analyzing and Summarizing News Articles using Large Language Models</div>', unsafe_allow_html=True)
st.markdown("---")


def run_pipeline(articles_input, topic_label):
    """Shared pipeline: chunk → embed → summarize → topics → synthesis"""
    prog = st.progress(0, text="✂️ Chunking articles with NLP...")
    from src.chunker import chunk_articles
    from src.llm_client import summarize_article, synthesize_summaries, detect_topics

    chunks = chunk_articles(articles_input)
    prog.progress(30, text="🧠 Building FAISS vector index...")
    vs = get_vector_store()
    vs.build(chunks)
    st.session_state.vector_store = vs

    prog.progress(50, text="📋 Generating AI summaries...")
    client = st.session_state.groq_client
    summaries = []
    for i, art in enumerate(articles_input):
        prog.progress(50 + int(30*(i+1)/len(articles_input)), text=f"📋 Summarizing {i+1}/{len(articles_input)}...")
        s = summarize_article(client, art.text, art.title)
        summaries.append({"title": art.title, "source": art.source, "url": art.url,
                          "summary": s, "word_count": art.word_count, "published": getattr(art,'published','')})

    prog.progress(83, text="🔗 Synthesizing across articles...")
    synthesis = synthesize_summaries(client, summaries) if len(summaries) > 1 else summaries[0]["summary"]

    prog.progress(93, text="🏷️ Detecting topics and themes...")
    combined = " ".join(a.text[:1500] for a in articles_input)
    topics_text = detect_topics(client, combined)

    prog.progress(100, text="✅ Done!")
    time.sleep(0.3)
    prog.empty()

    st.session_state.articles = articles_input
    st.session_state.summaries = summaries
    st.session_state.synthesis = synthesis
    st.session_state.topics_text = topics_text
    st.session_state.current_topic = topic_label
    st.session_state.processed = True
    st.session_state.chat_history = []
    st.rerun()


# ─── SEARCH PAGE ─────────────────────────────────────────────────────────────
if not st.session_state.processed:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#f8f7ff,#f3f0ff);border:2px solid #c4b5fd;
                border-radius:16px;padding:1.8rem 2rem;margin:0.5rem 0 1.5rem;text-align:center">
        <div style="font-size:1.15rem;font-weight:600;color:#4f46e5;margin-bottom:.3rem">
            🔍 Enter a topic to get started
        </div>
        <div style="font-size:.85rem;color:#7c3aed">
            NewsLens AI will automatically search, fetch, and analyze the latest news articles on your topic
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        topic_input = st.text_input("topic", placeholder="e.g.  Artificial Intelligence,  Climate Change,  India Economy 2025...",
                                    label_visibility="collapsed", key="topic_field")
    with col_btn:
        search_btn = st.button("🔍 Search", use_container_width=True)

    st.markdown("#### 💡 Trending Topics:")
    trending = ["Artificial Intelligence","Climate Change","India Economy",
                "Space Exploration","Electric Vehicles","Cybersecurity",
                "Ukraine War","Stock Market 2025","Cancer Research","Renewable Energy"]
    chip_cols = st.columns(5)
    for i, t in enumerate(trending):
        with chip_cols[i % 5]:
            if st.button(t, key=f"chip_{i}", use_container_width=True):
                st.session_state["_pt"] = t
                st.rerun()

    if "_pt" in st.session_state:
        topic_input = st.session_state.pop("_pt")
        search_btn = True

    # Manual URL fallback
    with st.expander("🔗 Advanced: Analyze specific URLs instead"):
        url_inputs = [st.text_input(f"URL {i}", key=f"url_{i}", placeholder="https://...") for i in range(1, 5)]
        url_btn = st.button("📥 Load URLs", key="url_submit")

    # Process: topic search
    if search_btn and topic_input.strip():
        if not api_key:
            st.warning("Please add your Groq API key in the sidebar first.")
        else:
            topic = topic_input.strip()
            prog = st.progress(0, text=f"🔍 Searching news for: {topic}...")
            from src.news_fetcher import fetch_articles_for_topic

            def upd(pct, msg): prog.progress(pct, text=msg)

            articles, errors = fetch_articles_for_topic(
                    topic,
                    max_articles=15,
                    newsapi_key=st.session_state.get("newsapi_key",""),
                    gnews_key=st.session_state.get("gnews_key",""),
                    progress_callback=upd,
                )
            for e in errors:
                st.warning(e)
            if not articles:
                st.error(f"Could not find articles for '{topic}'. Try a different topic.")
                prog.empty()
                st.stop()
            prog.empty()
            run_pipeline(articles, topic)

    # Process: URLs
    if url_btn:
        urls = [u.strip() for u in url_inputs if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        elif not api_key:
            st.warning("Please add your Groq API key first.")
        else:
            from src.scraper import scrape_multiple
            from src.news_fetcher import FetchedArticle
            raw, errors = scrape_multiple(urls)
            for e in errors: st.warning(e)
            if not raw:
                st.error("Could not extract content from any URLs.")
                st.stop()
            articles = [FetchedArticle(url=a.url, title=a.title, text=a.text, source=a.source,
                                       published=str(a.publish_date or ""), summary_snippet=a.text[:200],
                                       word_count=a.word_count, authors=a.authors) for a in raw]
            run_pipeline(articles, "Custom URLs")

    st.markdown("---")
    st.markdown("### ⚡ How NewsLens AI Works")
    hw_cols = st.columns(5)
    for col, icon, step, desc in zip(hw_cols,
        ["🔍","📡","✂️","🧠","💬"],
        ["1. Enter Topic","2. Fetch News","3. NLP Chunking","4. Vector Index","5. AI Analysis"],
        ["Type any subject","Live articles from Google News","Sliding-window text splits","FAISS semantic embeddings","Llama 3 answers & summarizes"]):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.8rem">{icon}</div>'
                        f'<div style="font-weight:600;font-size:.85rem;color:#1e293b;margin:.3rem 0">{step}</div>'
                        f'<div style="font-size:.75rem;color:#6b7280">{desc}</div></div>', unsafe_allow_html=True)

# ─── RESULTS PAGE ────────────────────────────────────────────────────────────
else:
    articles  = st.session_state.articles
    summaries = st.session_state.summaries
    topic     = st.session_state.current_topic

    # ── Always-visible search bar at top of results ──
    st.markdown("### 🔍 Search a new topic")
    rs_col1, rs_col2 = st.columns([5, 1])
    with rs_col1:
        new_topic = st.text_input("new_topic", placeholder="e.g. Climate Change, Gaza War, AI 2025...",
                                   label_visibility="collapsed", key="results_search")
    with rs_col2:
        new_search_btn = st.button("🔍 Search", key="results_search_btn", use_container_width=True)

    if new_search_btn and new_topic.strip():
        if not api_key:
            st.warning("Please add your Groq API key in the sidebar first.")
        else:
            prog = st.progress(0, text=f"🔍 Searching news for: {new_topic.strip()}...")
            from src.news_fetcher import fetch_articles_for_topic
            def upd2(pct, msg): prog.progress(pct, text=msg)
            new_articles, new_errors = fetch_articles_for_topic(
                new_topic.strip(), max_articles=15,
                newsapi_key=st.session_state.get("newsapi_key",""),
                gnews_key=st.session_state.get("gnews_key",""),
                progress_callback=upd2,
            )
            for e in new_errors: st.warning(e)
            if not new_articles:
                st.error(f"Could not find articles for '{new_topic}'. Try a different topic.")
                prog.empty()
            else:
                prog.empty()
                run_pipeline(new_articles, new_topic.strip())

    st.markdown("---")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
                border-radius:14px;padding:1rem 1.5rem;margin-bottom:1rem">
        <div style="font-size:.78rem;opacity:.85;text-transform:uppercase;letter-spacing:.08em">Currently analyzing</div>
        <div style="font-size:1.5rem;font-weight:700;margin-top:.1rem">🔍 {topic}</div>
        <div style="font-size:.8rem;opacity:.85;margin-top:.3rem">
            {len(articles)} articles · {sum(a.word_count for a in articles):,} words analyzed
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[
        (len(articles),"Articles"),
        (sum(a.word_count for a in articles),"Words Analyzed"),
        (len(set(a.source for a in articles)),"News Sources"),
        (len(st.session_state.vector_store.chunks) if st.session_state.vector_store else 0,"Chunks Indexed"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val:,}</div>'
                        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Synthesis & Summaries","💬 Q&A Chatbot","📊 Sentiment Analysis","🏷️ Topics & Insights"])

    # TAB 1
    with tab1:
        st.markdown("### 📡 Articles Found")
        ac = st.columns(2)
        for i, art in enumerate(articles):
            with ac[i%2]:
                st.markdown(f"""<div class="article-card">
                    <h4>{art.title[:78]}{'...' if len(art.title)>78 else ''}</h4>
                    <p><span class="source-badge">{art.source}</span>
                    {'📅 '+art.published[:16] if art.published else ''} · 📝 {art.word_count:,} words</p>
                    <p style="margin-top:4px"><a href="{art.url}" target="_blank"
                    style="color:#6366f1;font-size:.75rem">🔗 View article →</a></p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if len(summaries) > 1:
            st.markdown("### 🔗 AI Synthesis — The Full Picture")
            st.markdown('<div class="info-banner">✨ NewsLens AI has read all articles and combined them into one unified view</div>', unsafe_allow_html=True)
            st.markdown('<div class="synthesis-box">'+st.session_state.synthesis.replace("\n","<br>")+'</div>', unsafe_allow_html=True)
            st.markdown("")

        st.markdown("### 📄 Article-by-Article Summaries")
        for i, s in enumerate(summaries):
            with st.expander(f"**{i+1}. {s['title'][:72]}**  •  `{s['source']}`", expanded=(i==0)):
                st.markdown(f"🔗 [Read full article]({s['url']}) &nbsp;|&nbsp; 📝 {s['word_count']:,} words")
                st.markdown('<div class="summary-box">'+s["summary"].replace("\n","<br>")+'</div>', unsafe_allow_html=True)

    # TAB 2
    with tab2:
        st.markdown("### 💬 Ask Anything About These Articles")
        n_chunks = len(st.session_state.vector_store.chunks) if st.session_state.vector_store else 0
        st.markdown(f'<div class="info-banner">🧠 <strong>RAG-powered:</strong> Answers use semantic search over {n_chunks} indexed chunks from {len(articles)} articles on <strong>{topic}</strong>.</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            cls = "chat-user" if msg["role"]=="user" else "chat-assistant"
            icon = "🧑" if msg["role"]=="user" else "🤖"
            st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("#### 💡 Suggested questions:")
            suggestions = [
                f"What are the latest developments in {topic}?",
                f"Who are the key people or organizations in {topic}?",
                f"What are the main challenges related to {topic}?",
                "What is the overall conclusion across these articles?",
            ]
            sq1, sq2 = st.columns(2)
            for i,(col,q) in enumerate(zip([sq1,sq2,sq1,sq2], suggestions)):
                with col:
                    if st.button(q, key=f"sq_{i}"):
                        st.session_state["_pq"] = q
                        st.rerun()

        with st.form("chatform", clear_on_submit=True):
            pf = st.session_state.pop("_pq", "")
            uq = st.text_input("q", value=pf, placeholder=f"Ask about {topic}...", label_visibility="collapsed")
            sent = st.form_submit_button("Send ➤")

        if sent and uq.strip():
            st.session_state.chat_history.append({"role":"user","content":uq})
            with st.spinner("Searching and generating answer..."):
                vs = st.session_state.vector_store
                results = vs.search(uq, top_k=5)
                ctx = [r[0].text for r in results]
                srcs = [r[0].source_title[:40] for r in results]
                from src.llm_client import answer_question
                ans = answer_question(st.session_state.groq_client, uq, ctx, srcs)
            st.session_state.chat_history.append({"role":"assistant","content":ans})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # TAB 3
    with tab3:
        st.markdown("### 📊 Sentiment Analysis")
        from src.sentiment import analyze_articles_sentiment, analyze_sentence_sentiments
        sr = analyze_articles_sentiment(articles)
        df = pd.DataFrame([{
            "Article": r["title"][:35]+"...", "Source": r["source"],
            "Score": r["sentiment"].compound, "Label": r["sentiment"].label,
            "Positive %": round(r["sentiment"].positive*100,1),
            "Negative %": round(r["sentiment"].negative*100,1),
            "Neutral %":  round(r["sentiment"].neutral*100,1),
        } for r in sr])
        cm = {"POSITIVE":"#16a34a","NEGATIVE":"#dc2626","NEUTRAL":"#d97706"}

        cc, cd = st.columns([3,2])
        with cc:
            fig = go.Figure()
            for _, row in df.iterrows():
                fig.add_trace(go.Bar(x=[row["Article"]], y=[row["Score"]],
                    marker_color=cm.get(row["Label"],"#6366f1"), showlegend=False,
                    hovertemplate=f"<b>{row['Article']}</b><br>{row['Label']}: {row['Score']:.3f}<extra></extra>"))
            fig.add_hline(y=0.05, line_dash="dash", line_color="#16a34a", opacity=0.4)
            fig.add_hline(y=-0.05, line_dash="dash", line_color="#dc2626", opacity=0.4)
            fig.update_layout(title=f"Sentiment — {topic}", yaxis_title="Score (-1 to +1)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with cd:
            st.markdown("#### Per-Article")
            for r in sr:
                s = r["sentiment"]
                bg = {"POSITIVE":"#f0fdf4","NEGATIVE":"#fef2f2","NEUTRAL":"#fffbeb"}.get(s.label,"#f8fafc")
                bd = cm.get(s.label,"#6b7280")
                st.markdown(f'<div style="background:{bg};border-left:3px solid {bd};border-radius:8px;padding:.6rem .9rem;margin-bottom:.4rem">'
                    f'<div style="font-weight:600;font-size:.8rem">{r["title"][:48]}...</div>'
                    f'<div style="font-size:.75rem;margin-top:2px">{s.emoji} <strong>{s.label}</strong> {s.compound:+.3f}</div>'
                    f'<div style="font-size:.7rem;color:#6b7280">🟢{s.positive:.0%} 🔴{s.negative:.0%} 🟡{s.neutral:.0%}</div>'
                    f'</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        for name, col_key, color in [("Positive","Positive %","#16a34a"),("Negative","Negative %","#dc2626"),("Neutral","Neutral %","#d97706")]:
            fig2.add_trace(go.Bar(name=name, x=df["Article"], y=df[col_key], marker_color=color))
        fig2.update_layout(barmode="stack", yaxis_title="Percentage (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10,b=20), legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🔥 Most Emotionally Charged Sentences")
        sel = st.selectbox("Select article", range(len(articles)), format_func=lambda i: articles[i].title[:60])
        charged = analyze_sentence_sentiments(articles[sel].text)
        for item in charged:
            bg = {"POSITIVE":"#dcfce7","NEGATIVE":"#fee2e2","NEUTRAL":"#fef9c3"}.get(item["label"],"#f1f5f9")
            bd = cm.get(item["label"],"#6b7280")
            st.markdown(f'<div style="background:{bg};border-left:3px solid {bd};padding:.5rem .9rem;border-radius:6px;margin-bottom:.4rem;font-size:.85rem">'
                        f'{item["emoji"]} <strong>{item["label"]}</strong> ({item["compound"]:+.3f}) — {item["sentence"]}</div>', unsafe_allow_html=True)

    # TAB 4
    with tab4:
        st.markdown("### 🏷️ Topic Detection & Thematic Insights")
        st.markdown('<div class="info-banner">🤖 Detected by <strong>Llama 3 (Groq)</strong> from the combined article content</div>', unsafe_allow_html=True)
        ct, cr = st.columns([5,1])
        with cr:
            if st.button("🔄 Re-detect"):
                with st.spinner("Re-detecting..."):
                    from src.llm_client import detect_topics
                    combined = " ".join(a.text[:1500] for a in articles)
                    st.session_state.topics_text = detect_topics(st.session_state.groq_client, combined)
                st.rerun()
        with ct:
            st.markdown('<div class="summary-box">'+st.session_state.topics_text.replace("\n","<br>")+'</div>', unsafe_allow_html=True)

        st.markdown("---")
        wc_df = pd.DataFrame({"Article":[a.title[:38]+"..." for a in articles],
                               "Words":[a.word_count for a in articles],"Source":[a.source for a in articles]})
        fig3 = px.bar(wc_df, x="Article", y="Words", color="Source", title="Article Length by Source",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=290, margin=dict(t=40,b=20))
        st.plotly_chart(fig3, use_container_width=True)

        meta_df = pd.DataFrame([{"Title":a.title[:55]+("..." if len(a.title)>55 else ""),
            "Source":a.source,"Words":a.word_count,
            "Published":a.published[:16] if a.published else "N/A",
            "Authors":", ".join(a.authors[:2]) if a.authors else "N/A","URL":a.url} for a in articles])
        st.dataframe(meta_df, use_container_width=True, hide_index=True)