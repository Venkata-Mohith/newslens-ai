"""
llm_client.py - Groq API wrapper for fast, free LLM inference.
Uses Llama 3 8B by default (free tier at console.groq.com).
"""

import os
from groq import Groq


MODEL = "llama-3.1-8b-instant"          # Fast, free, great quality
FALLBACK_MODEL = "llama-3.1-70b-versatile"


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def chat(
    client: Groq,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Send a chat message and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if "model" in str(e).lower():
            # Try fallback model
            response = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        raise e


def summarize_article(client: Groq, article_text: str, title: str) -> str:
    """Generate a concise summary of a single article."""
    system = (
        "You are an expert news analyst. Summarize news articles clearly and concisely. "
        "Always include: (1) the main event/topic, (2) key facts and figures, "
        "(3) who is involved, (4) why it matters. Use bullet points for key facts."
    )
    user = (
        f"Article Title: {title}\n\n"
        f"Article Text:\n{article_text[:4000]}\n\n"
        "Please provide a concise, structured summary (150-250 words)."
    )
    return chat(client, system, user, temperature=0.2, max_tokens=512)


def synthesize_summaries(client: Groq, summaries: list[dict]) -> str:
    """Create a unified cross-article synthesis."""
    system = (
        "You are an expert news editor. Given summaries of multiple articles on related topics, "
        "synthesize them into one comprehensive overview. Highlight common themes, "
        "differing perspectives, and key takeaways."
    )
    combined = "\n\n---\n\n".join(
        f"Source: {s['source']}\nTitle: {s['title']}\nSummary: {s['summary']}"
        for s in summaries
    )
    user = (
        f"Here are summaries from {len(summaries)} news articles:\n\n{combined}\n\n"
        "Provide a unified synthesis (200-300 words) covering the full picture."
    )
    return chat(client, system, user, temperature=0.3, max_tokens=700)


def answer_question(client: Groq, question: str, context_chunks: list[str], sources: list[str]) -> str:
    """Answer a user question using retrieved context chunks (RAG)."""
    system = (
        "You are a helpful news research assistant. Answer questions accurately based ONLY on "
        "the provided news article excerpts. If the answer isn't in the context, say so clearly. "
        "Always cite which source(s) your answer comes from."
    )
    context = "\n\n".join(
        f"[Excerpt {i+1} from {src}]:\n{chunk}"
        for i, (chunk, src) in enumerate(zip(context_chunks, sources))
    )
    user = (
        f"Context from news articles:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based on the context above. Cite your sources."
    )
    return chat(client, system, user, temperature=0.2, max_tokens=600)


def detect_topics(client: Groq, combined_text: str) -> str:
    """Detect main topics/themes across all articles."""
    system = (
        "You are a topic modeling expert. Identify the main topics, themes, and entities "
        "in news articles. Be specific and structured."
    )
    user = (
        f"Analyze this combined news text and identify:\n"
        f"1. Main topics (3-5 topics)\n"
        f"2. Key entities (people, organizations, places)\n"
        f"3. Key themes and trends\n\n"
        f"Text:\n{combined_text[:5000]}\n\n"
        f"Format your response with clear headers for each section."
    )
    return chat(client, system, user, temperature=0.2, max_tokens=600)