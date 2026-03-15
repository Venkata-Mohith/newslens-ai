"""
sentiment.py - Sentiment analysis using VADER (free, no API needed).
Analyzes sentiment at article level and sentence level.
"""

import re
from dataclasses import dataclass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


@dataclass
class SentimentResult:
    label: str          # POSITIVE / NEGATIVE / NEUTRAL
    score: float        # -1.0 to 1.0
    positive: float     # 0.0 to 1.0
    negative: float     # 0.0 to 1.0
    neutral: float      # 0.0 to 1.0
    compound: float     # VADER compound score
    emoji: str          # Visual indicator


def _score_to_label(compound: float) -> tuple[str, str]:
    if compound >= 0.05:
        return "POSITIVE", "🟢"
    elif compound <= -0.05:
        return "NEGATIVE", "🔴"
    else:
        return "NEUTRAL", "🟡"


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze overall sentiment of a text block."""
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        label, emoji = _score_to_label(scores["compound"])
        return SentimentResult(
            label=label,
            score=scores["compound"],
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            compound=scores["compound"],
            emoji=emoji,
        )
    else:
        # Very basic fallback
        positive_words = ["good", "great", "excellent", "positive", "success", "win", "growth", "improve"]
        negative_words = ["bad", "poor", "negative", "fail", "loss", "decline", "crisis", "problem"]
        text_lower = text.lower()
        pos = sum(text_lower.count(w) for w in positive_words)
        neg = sum(text_lower.count(w) for w in negative_words)
        total = pos + neg + 1
        compound = (pos - neg) / total
        label, emoji = _score_to_label(compound)
        return SentimentResult(
            label=label, score=compound,
            positive=pos/total, negative=neg/total, neutral=0.5,
            compound=compound, emoji=emoji,
        )


def analyze_sentence_sentiments(text: str, top_n: int = 6) -> list[dict]:
    """
    Break text into sentences and analyze each.
    Returns top N most emotionally charged sentences.
    """
    if not VADER_AVAILABLE:
        return []

    analyzer = SentimentIntensityAnalyzer()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        scores = analyzer.polarity_scores(sent)
        label, emoji = _score_to_label(scores["compound"])
        results.append({
            "sentence": sent[:200],
            "label": label,
            "emoji": emoji,
            "compound": scores["compound"],
        })

    # Sort by absolute compound score (most charged first)
    results.sort(key=lambda x: abs(x["compound"]), reverse=True)
    return results[:top_n]


def analyze_articles_sentiment(articles: list) -> list[dict]:
    """Analyze sentiment for each article and return comparison list."""
    results = []
    for article in articles:
        sentiment = analyze_sentiment(article.text)
        results.append({
            "title": article.title[:60] + ("..." if len(article.title) > 60 else ""),
            "source": article.source,
            "url": article.url,
            "sentiment": sentiment,
            "word_count": article.word_count,
        })
    return results
