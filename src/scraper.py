"""
scraper.py - Extracts article text, title, and metadata from news URLs
Uses newspaper3k with BeautifulSoup as fallback
"""

import requests
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
from typing import Optional

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


@dataclass
class ArticleData:
    url: str
    title: str
    text: str
    authors: list
    publish_date: Optional[str]
    source: str
    word_count: int


def _extract_domain(url: str) -> str:
    """Extract domain name from URL for source labeling."""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else "Unknown"


def _fallback_scrape(url: str) -> dict:
    """BeautifulSoup fallback when newspaper3k fails."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else (soup.title.get_text(strip=True) if soup.title else "Untitled")

    # Try to find article body
    article_tag = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"article|content|body|story", re.I))
    if article_tag:
        paragraphs = article_tag.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
    return {"title": title_text, "text": text, "authors": [], "publish_date": None}


def scrape_article(url: str) -> ArticleData:
    """
    Scrape an article from a URL.
    Returns ArticleData with title, text, authors, and metadata.
    """
    url = url.strip()
    domain = _extract_domain(url)

    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()

            if len(article.text.strip()) < 100:
                raise ValueError("Too short, trying fallback")

            pub_date = str(article.publish_date.date()) if article.publish_date else None
            return ArticleData(
                url=url,
                title=article.title or "Untitled",
                text=article.text.strip(),
                authors=list(article.authors),
                publish_date=pub_date,
                source=domain,
                word_count=len(article.text.split()),
            )
        except Exception:
            pass  # Fall through to BeautifulSoup fallback

    # Fallback
    data = _fallback_scrape(url)
    return ArticleData(
        url=url,
        title=data["title"],
        text=data["text"].strip(),
        authors=data["authors"],
        publish_date=data["publish_date"],
        source=domain,
        word_count=len(data["text"].split()),
    )


def scrape_multiple(urls: list[str]) -> tuple[list[ArticleData], list[str]]:
    """
    Scrape multiple URLs.
    Returns (successful_articles, error_messages)
    """
    articles = []
    errors = []
    for url in urls:
        if not url.strip():
            continue
        try:
            article = scrape_article(url)
            if len(article.text) < 100:
                errors.append(f"⚠️ {url}: Could not extract enough text.")
            else:
                articles.append(article)
        except Exception as e:
            errors.append(f"❌ {url}: {str(e)}")
    return articles, errors
