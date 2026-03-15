"""
news_fetcher.py - Multi-source news fetcher.
Sources (all free):
  1. NewsAPI.org       — accurate keyword search, up to 100 req/day free
  2. GNews API         — another free news API, 100 req/day
  3. Google News RSS   — unlimited, broad
  4. Bing News RSS     — unlimited, alternate angles
  5. The Guardian RSS  — unlimited, quality journalism
  6. BBC / Reuters / AP / NPR RSS — major wire services
  7. Topic-specific RSS feeds     — WHO, TechCrunch, ToI, etc.
  8. Reddit r/news, r/worldnews   — community-curated
"""

import os
import socket
import requests
import feedparser
socket.setdefaulttimeout(10)  # global socket timeout so no RSS feed hangs forever
import re
import time
from dataclasses import dataclass
from typing import Optional, Callable
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


@dataclass
class FetchedArticle:
    url: str
    title: str
    text: str
    source: str
    published: str
    summary_snippet: str
    word_count: int
    authors: list
    relevance_score: float = 0.0   # how relevant to the query (0-1)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def _relevance_score(text: str, query_words: list[str]) -> float:
    """Score 0-1 how relevant a text is to query words."""
    if not query_words or not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return round(hits / len(query_words), 3)


def _is_relevant(title: str, snippet: str, query_words: list[str], threshold: float = 0.25) -> bool:
    """Return True only if the article is sufficiently relevant to the topic."""
    combined = (title + " " + snippet).lower()
    score = _relevance_score(combined, query_words)
    return score >= threshold


# ── API-based Sources (most accurate) ────────────────────────────────────────

def _newsapi_fetch(query: str, api_key: str, n: int = 20) -> list[dict]:
    """
    NewsAPI.org — free tier: 100 req/day
    Sign up at https://newsapi.org/register
    Returns articles sorted by relevancy.
    """
    if not api_key:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(n, 20),
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        results = []
        for a in data.get("articles", []):
            if a.get("url") and a.get("title") and "[Removed]" not in a.get("title",""):
                results.append({
                    "title": a.get("title", "").strip(),
                    "url": a.get("url", "").strip(),
                    "published": a.get("publishedAt", ""),
                    "snippet": (a.get("description") or a.get("content") or "")[:400],
                    "source": a.get("source", {}).get("name", _extract_domain(a.get("url",""))),
                })
        return results
    except Exception:
        return []


def _gnews_fetch(query: str, api_key: str, n: int = 10) -> list[dict]:
    """
    GNews API — free tier: 100 req/day
    Sign up at https://gnews.io
    """
    if not api_key:
        return []
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "token": api_key,
            "lang": "en",
            "max": min(n, 10),
            "sortby": "relevance",
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        results = []
        for a in data.get("articles", []):
            if a.get("url") and a.get("title"):
                results.append({
                    "title": a.get("title", "").strip(),
                    "url": a.get("url", "").strip(),
                    "published": a.get("publishedAt", ""),
                    "snippet": (a.get("description") or "")[:400],
                    "source": a.get("source", {}).get("name", _extract_domain(a.get("url",""))),
                })
        return results
    except Exception:
        return []


# ── Free RSS Sources ──────────────────────────────────────────────────────────

def _google_news(query: str, n: int = 20) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        return [_entry_to_dict(e) for e in feed.entries[:n]]
    except Exception:
        return []


def _bing_news(query: str, n: int = 15) -> list[dict]:
    url = f"https://www.bing.com/news/search?q={quote_plus(query)}&format=RSS"
    try:
        feed = feedparser.parse(url)
        return [_entry_to_dict(e) for e in feed.entries[:n]]
    except Exception:
        return []


def _guardian_rss(query: str, n: int = 10) -> list[dict]:
    url = f"https://www.theguardian.com/search?q={quote_plus(query)}&format=rss"
    try:
        feed = feedparser.parse(url)
        return [_entry_to_dict(e) for e in feed.entries[:n]]
    except Exception:
        return []


def _reddit_news(query: str, n: int = 8) -> list[dict]:
    results = []
    for sub in ["news", "worldnews", "science", "technology", "health"]:
        url = f"https://www.reddit.com/r/{sub}/search.rss?q={quote_plus(query)}&restrict_sr=1&sort=relevance&t=month"
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "NewsLensAI/1.0"})
            for e in feed.entries[:4]:
                link = e.get("link", "")
                if "reddit.com" not in link:
                    results.append(_entry_to_dict(e))
        except Exception:
            continue
    return results[:n]


def _bbc_rss(query: str) -> list[dict]:
    try:
        feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
        return [_entry_to_dict(e) for e in feed.entries[:15]]
    except Exception:
        return []


def _reuters_rss(query: str) -> list[dict]:
    try:
        feed = feedparser.parse("https://feeds.reuters.com/reuters/topNews")
        return [_entry_to_dict(e) for e in feed.entries[:15]]
    except Exception:
        return []


def _ap_rss(query: str) -> list[dict]:
    try:
        feed = feedparser.parse("https://apnews.com/rss")
        return [_entry_to_dict(e) for e in feed.entries[:15]]
    except Exception:
        return []


def _npr_rss(query: str) -> list[dict]:
    try:
        feed = feedparser.parse("https://feeds.npr.org/1001/rss.xml")
        return [_entry_to_dict(e) for e in feed.entries[:10]]
    except Exception:
        return []


TOPIC_FEEDS = {
    "covid": [
        "https://www.who.int/rss-feeds/news-english.xml",
        "https://tools.cdc.gov/api/v2/resources/media/132608.rss",
        "https://www.sciencedaily.com/rss/health_medicine/covid-19.xml",
    ],
    "health": [
        "https://www.who.int/rss-feeds/news-english.xml",
        "https://www.sciencedaily.com/rss/health_medicine.xml",
        "https://feeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
    ],
    "ai": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://www.wired.com/feed/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
    "technology": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://www.wired.com/feed/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
    "climate": [
        "https://insideclimatenews.org/feed/",
        "https://www.sciencedaily.com/rss/earth_climate/climate.xml",
    ],
    "science": [
        "https://www.sciencedaily.com/rss/all.xml",
    ],
    "india": [
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://www.thehindu.com/news/feeder/default.rss",
    ],
    "finance": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
    ],
    "business": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
    ],
    "space": [
        "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "https://www.sciencedaily.com/rss/space_time.xml",
    ],
    "cancer": [
        "https://www.cancer.gov/news-events/cancer-currents-blog/feed",
        "https://www.sciencedaily.com/rss/health_medicine/cancer.xml",
    ],
    "electric": [
        "https://electrek.co/feed/",
    ],
    "ukraine": [
        "https://www.theguardian.com/world/ukraine/rss",
    ],
    "cyber": [
        "https://feeds.feedburner.com/TheHackersNews",
        "https://www.darkreading.com/rss.xml",
    ],
}


def _get_topic_feeds(query: str, query_words: list[str], n: int = 12) -> list[dict]:
    q_lower = query.lower()
    matched_urls = []
    for keyword, feeds in TOPIC_FEEDS.items():
        if keyword in q_lower:
            matched_urls.extend(feeds)

    results = []
    for feed_url in matched_urls[:5]:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "")
                snippet = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:300]
                if _is_relevant(title, snippet, query_words, threshold=0.2):
                    results.append(_entry_to_dict(entry))
        except Exception:
            continue
    return results[:n]


def _entry_to_dict(entry) -> dict:
    snippet = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:400]
    return {
        "title": entry.get("title", "").strip(),
        "url": entry.get("link", "").strip(),
        "published": entry.get("published", ""),
        "snippet": snippet,
        "source": (entry.get("source", {}) or {}).get("title", "") or _extract_domain(entry.get("link", "")),
    }


# ── Article Scraper ───────────────────────────────────────────────────────────

def _scrape_full_text(url: str) -> tuple[str, list]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    if NEWSPAPER_AVAILABLE:
        try:
            art = Article(url)
            art.download()
            art.parse()
            if len(art.text.strip()) > 200:
                return art.text.strip(), list(art.authors)
        except Exception:
            pass
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        container = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_=re.compile(r"article|content|body|story|post", re.I))
        )
        paragraphs = (container or soup).find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
        return text.strip(), []
    except Exception:
        return "", []


def _scrape_one(meta: dict, query_words: list[str]) -> Optional[FetchedArticle]:
    url = meta.get("url", "")
    if not url or any(s in url for s in ["google.com/search","bing.com/search","reddit.com/r/"]):
        return None
    try:
        text, authors = _scrape_full_text(url)
        if len(text) < 150:
            text = meta.get("snippet", "")
        if len(text) < 80:
            return None
        # Final relevance check on actual article text
        title_snippet = meta.get("title","") + " " + meta.get("snippet","")
        rscore = _relevance_score(title_snippet + " " + text[:500], query_words)
        if rscore < 0.15:   # hard cutoff — not related enough
            return None
        return FetchedArticle(
            url=url,
            title=meta["title"],
            text=text,
            source=meta.get("source") or _extract_domain(url),
            published=meta.get("published", ""),
            summary_snippet=meta.get("snippet", ""),
            word_count=len(text.split()),
            authors=authors,
            relevance_score=rscore,
        )
    except Exception:
        return None


# ── Main Public Function ──────────────────────────────────────────────────────

def fetch_articles_for_topic(
    topic: str,
    max_articles: int = 15,
    newsapi_key: str = "",
    gnews_key: str = "",
    progress_callback: Optional[Callable] = None,
) -> tuple[list[FetchedArticle], list[str]]:
    """
    Fetch up to max_articles relevant news articles on a topic.
    Uses NewsAPI + GNews (if keys given) + multiple free RSS sources.
    Filters irrelevant articles by relevance score.
    """
    query_words = [w for w in re.sub(r'[^\w\s]', '', topic.lower()).split() if len(w) > 2]

    if progress_callback:
        progress_callback(5, f"🔍 Searching across all news sources for: **{topic}**...")

    # ── Phase 1: Collect metadata from all sources in parallel ────────────────
    all_meta: list[dict] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    def _add_unique(items: list[dict]):
        for item in items:
            url = item.get("url", "").strip()
            title = item.get("title", "").strip().lower()[:80]
            if not url or not title or url in seen_urls or title in seen_titles:
                continue
            if any(s in url for s in ["google.com/search","bing.com/search"]):
                continue
            # Relevance pre-filter on title+snippet
            if not _is_relevant(item.get("title",""), item.get("snippet",""), query_words, threshold=0.15):
                return
            seen_urls.add(url)
            seen_titles.add(title)
            all_meta.append(item)

    if progress_callback:
        progress_callback(10, "📡 Querying NewsAPI, GNews, Google News, Bing, Guardian, BBC, Reuters, AP...")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        # API-based (most accurate — run first)
        if newsapi_key:
            futures[ex.submit(_newsapi_fetch, topic, newsapi_key, 20)] = "NewsAPI"
        if gnews_key:
            futures[ex.submit(_gnews_fetch, topic, gnews_key, 10)] = "GNews"
        # RSS-based (always run)
        futures[ex.submit(_google_news, topic, 20)]        = "Google News"
        futures[ex.submit(_bing_news, topic, 15)]          = "Bing News"
        futures[ex.submit(_guardian_rss, topic, 10)]       = "The Guardian"
        futures[ex.submit(_reddit_news, topic, 8)]         = "Reddit"
        futures[ex.submit(_get_topic_feeds, topic, query_words, 12)] = "Topic Feeds"
        # Broad wire services filtered by topic
        futures[ex.submit(_bbc_rss, topic)]      = "BBC"
        futures[ex.submit(_reuters_rss, topic)]  = "Reuters"
        futures[ex.submit(_ap_rss, topic)]       = "AP"
        futures[ex.submit(_npr_rss, topic)]      = "NPR"

        for future in as_completed(futures):
            try:
                _add_unique(future.result(timeout=12))
            except Exception:
                pass  # skip slow/failed sources gracefully

    if progress_callback:
        progress_callback(30, f"📰 {len(all_meta)} relevant candidates found. Fetching full content...")

    if not all_meta:
        return [], [f"❌ No relevant articles found for '{topic}'. Try rephrasing."]

    # ── Phase 2: Sort candidates by relevance before scraping ─────────────────
    all_meta.sort(
        key=lambda m: _relevance_score(m.get("title","")+" "+m.get("snippet",""), query_words),
        reverse=True,
    )

    # Allow up to 3 articles per domain
    domain_count: dict[str, int] = {}
    filtered_meta: list[dict] = []
    for m in all_meta:
        d = _extract_domain(m.get("url", ""))
        if domain_count.get(d, 0) < 3:
            domain_count[d] = domain_count.get(d, 0) + 1
            filtered_meta.append(m)
        if len(filtered_meta) >= max_articles + 12:
            break

    # ── Phase 3: Parallel full-text scraping ──────────────────────────────────
    articles: list[FetchedArticle] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_meta = {ex.submit(_scrape_one, m, query_words): m for m in filtered_meta}
        done = 0
        for future in as_completed(future_to_meta):
            done += 1
            if progress_callback:
                pct = 30 + int(55 * done / max(len(filtered_meta), 1))
                progress_callback(min(pct, 85), f"📄 {len(articles)} articles loaded... ({done}/{len(filtered_meta)} processed)")
            try:
                result = future.result(timeout=15)
                if result:
                    articles.append(result)
                else:
                    meta = future_to_meta[future]
                    errors.append(f"⚠️ Skipped: {meta.get('title','')[:50]}")
            except Exception as e:
                meta = future_to_meta[future]
                errors.append(f"⚠️ Failed: {meta.get('title','')[:40]}")

            if len(articles) >= max_articles:
                break

    # Sort final list by relevance score desc, then word count
    articles.sort(key=lambda a: (a.relevance_score, a.word_count), reverse=True)
    articles = articles[:max_articles]

    if not articles:
        errors.append(f"❌ Could not load any articles for '{topic}'. Try a broader topic.")

    if progress_callback:
        progress_callback(88, f"✅ Loaded {len(articles)} relevant articles!")

    return articles, errors