# Asknews.py
from __future__ import annotations

import inspect
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests
import streamlit as st
from pydantic import BaseModel, ConfigDict

# pip install asknews
from asknews_sdk import AskNewsSDK


# ============================================================
# CONFIG
# ============================================================
ASKNEWS_API_KEY = "ank_tIpMbXiY2OSUWCU1RvO9IJkFbqVRUMO5HmNg2AGSjz"  # <- paste your ank_... key here
OPENROUTER_API_KEY = "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"


# ============================================================
# Simple data classes (no external deps)
# ============================================================
class ArticleDoc(BaseModel):
    """Canonical article representation for topic clustering."""

    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    headline: Optional[str] = None
    name: Optional[str] = None

    summary: Optional[str] = None
    snippet: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    text: Optional[str] = None

    url: Optional[str] = None
    link: Optional[str] = None

    published_at: Optional[Any] = None

    @classmethod
    def from_any(cls, item: Any) -> "ArticleDoc":
        if isinstance(item, cls):
            return item

        data: Dict[str, Any] = {}
        if isinstance(item, dict):
            data = item
        else:
            for field_name in (
                "title",
                "headline",
                "name",
                "summary",
                "snippet",
                "description",
                "content",
                "text",
                "url",
                "link",
                "published_at",
            ):
                if hasattr(item, field_name):
                    data[field_name] = getattr(item, field_name)
        return cls.model_validate(data)

    def best_title(self) -> str:
        for v in (self.title, self.headline, self.name):
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def best_summary(self) -> str:
        for v in (self.summary, self.snippet, self.description, self.content, self.text):
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def best_url(self) -> str:
        for v in (self.url, self.link):
            if isinstance(v, str) and v.startswith("http"):
                return v
        return ""

    def as_corpus_text(self) -> str:
        t = self.best_title()
        s = self.best_summary()
        if t and s:
            return f"{t}\n{s}"
        return t or s


@dataclass
class Topic:
    topic_id: str
    mode: str
    title: str
    query: str
    summary: str = ""
    key_takeaways: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    support_count: int = 0

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Low-level utilities
# ============================================================
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "from","as","is","are","was","were","be","been","being","it","this","that","these","those","their","its","his",
    "her","they","them","we","us","you","your","i","me","my","our","not","no","yes","do","does","did","doing",
    "can","could","may","might","will","would","shall","should","must","about","into","over","after","before","up",
    "down","out","off","again","more","most","some","such","only","own","same","so","than","too","very"
}


def now_utc_ts() -> int:
    return int(time.time())


def safe_slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] or "topic"


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
    return toks


def filtered_kwargs(fn, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs by the installed SDK method signature."""
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}


def extract_api_error_details(e: Exception) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in ("code", "status_code", "detail", "message"):
        if hasattr(e, attr):
            out[attr] = getattr(e, attr)

    resp = getattr(e, "response", None)
    if resp is not None:
        out["response.status_code"] = getattr(resp, "status_code", None)
        # requests-like response
        try:
            if hasattr(resp, "json"):
                out["response.json"] = resp.json()
        except Exception:
            pass
        try:
            if hasattr(resp, "text"):
                out["response.text"] = resp.text
        except Exception:
            pass
    return out


def refine_topics_with_llm(
    topics: List[Topic],
    instructions: str,
    model_name: str,
    desired_count: int,
) -> List[Topic]:
    if not instructions.strip():
        return topics[:desired_count]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un éditeur de sujets d'actualité. Garde les champs JSON: "
                    "topic_id, mode, title, query, summary, key_takeaways, source_urls, support_count. "
                    "Renvoie uniquement un tableau JSON compact sans texte additionnel."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Sujets actuels:\n"
                    f"{json.dumps([t.model_dump() for t in topics], ensure_ascii=False)}\n\n"
                    f"Instructions de modification: {instructions}\n"
                    f"Nombre maximum de sujets: {desired_count}."
                ),
            },
        ],
        "max_tokens": 800,
        "temperature": 0.4,
    }

    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        # Extract JSON array from possible markdown fences
        match = re.search(r"\[.*\]", content, re.DOTALL)
        json_text = match.group(0) if match else content
        parsed = json.loads(json_text)
        refined: List[Topic] = []
        if isinstance(parsed, list):
            for item in parsed[:desired_count]:
                if not isinstance(item, dict):
                    continue
                refined.append(
                    Topic(
                        topic_id=str(item.get("topic_id") or safe_slug(item.get("title", "topic"))),
                        mode=str(item.get("mode") or "news"),
                        title=str(item.get("title") or "Sujet"),
                        query=str(item.get("query") or ""),
                        summary=str(item.get("summary") or ""),
                        key_takeaways=[str(x) for x in item.get("key_takeaways", []) if isinstance(x, str)],
                        source_urls=[str(x) for x in item.get("source_urls", []) if isinstance(x, str)],
                        support_count=int(item.get("support_count") or 0),
                    )
                )
        return refined or topics[:desired_count]
    except Exception:
        return topics[:desired_count]


def is_invalid_permissions(e: Exception) -> bool:
    code = getattr(e, "code", None)
    resp = getattr(e, "response", None)
    status = getattr(resp, "status_code", None) if resp is not None else getattr(e, "status_code", None)
    detail = getattr(e, "detail", "") or ""
    return (code == 403000) or (status == 403 and "Invalid Permissions" in str(detail))


def normalize_to_items(resp: Any) -> List[Any]:
    """Return items (articles or stories) without converting to dicts."""
    if resp is None:
        return []

    if isinstance(resp, tuple):
        if len(resp) == 2 and isinstance(resp[1], (list, tuple)):
            return list(resp[1])
        return list(resp)

    if isinstance(resp, list):
        return resp

    for attr in ("articles", "stories", "results", "data", "items"):
        if hasattr(resp, attr):
            v = getattr(resp, attr)
            if isinstance(v, (list, tuple)):
                return list(v)

    if isinstance(resp, dict):
        for key in ("articles", "stories", "results", "data", "items"):
            v = resp.get(key)
            if isinstance(v, list):
                return v
        return [resp]

    # last resort: try to iterate
    try:
        return list(resp)  # type: ignore[arg-type]
    except Exception:
        return [resp]


# ============================================================
# AskNews client + endpoint callers
# ============================================================
@st.cache_resource
def get_client() -> AskNewsSDK:
    if not ASKNEWS_API_KEY.strip():
        raise RuntimeError('ASKNEWS_API_KEY is empty. Paste your key into: ASKNEWS_API_KEY = "ank_..."')
    return AskNewsSDK(api_key=ASKNEWS_API_KEY.strip())


def call_stories_search(ask: AskNewsSDK, params: Dict[str, Any]) -> Any:
    stories_client = getattr(ask, "stories", None)
    if stories_client is None:
        raise RuntimeError("AskNewsSDK has no 'stories' client. Check your installed asknews_sdk version.")
    fn = getattr(stories_client, "search_stories", None)
    if not callable(fn):
        # Try a couple of alternates, just in case.
        for alt in ("search", "get_stories", "list_stories"):
            fn = getattr(stories_client, alt, None)
            if callable(fn):
                break
    if not callable(fn):
        raise RuntimeError("No callable stories search method found on ask.stories.")
    return fn(**filtered_kwargs(fn, params))


def call_news_search(ask: AskNewsSDK, params: Dict[str, Any]) -> Any:
    news_client = getattr(ask, "news", None)
    if news_client is None:
        raise RuntimeError("AskNewsSDK has no 'news' client. Check your installed asknews_sdk version.")
    fn = getattr(news_client, "search_news", None)
    if not callable(fn):
        # Try a couple of alternates, just in case.
        for alt in ("search", "get_news", "list_news", "search_articles"):
            fn = getattr(news_client, alt, None)
            if callable(fn):
                break
    if not callable(fn):
        raise RuntimeError("No callable news search method found on ask.news.")
    return fn(**filtered_kwargs(fn, params))


@st.cache_data(ttl=300)
def cached_stories(params: Dict[str, Any]) -> List[Any]:
    ask = get_client()
    resp = call_stories_search(ask, params)
    return normalize_to_items(resp)


@st.cache_data(ttl=300)
def cached_news(params: Dict[str, Any]) -> List[Any]:
    ask = get_client()
    resp = call_news_search(ask, params)
    return normalize_to_items(resp)


# ============================================================
# STORIES -> Topic conversion (best quality, requires permission)
# ============================================================
def stories_to_topics(stories: Sequence[Any]) -> List[Topic]:
    """
    Stories objects are SDK-dependent; we keep this conversion conservative.
    If Stories are available, we prefer their own headline/story/takeaways if present.
    """
    topics: List[Topic] = []

    for s in stories:
        # Try to access common attributes without dict conversion
        updates = getattr(s, "updates", None)
        latest_update = updates[0] if isinstance(updates, list) and updates else None

        headline = getattr(latest_update, "headline", None) if latest_update is not None else None
        story_text = getattr(latest_update, "story", None) if latest_update is not None else None
        takeaways = getattr(latest_update, "key_takeaways", None) if latest_update is not None else None
        n_articles = getattr(latest_update, "n_articles", None) if latest_update is not None else None

        title = ""
        for v in (headline, getattr(s, "title", None), getattr(s, "name", None)):
            if isinstance(v, str) and v.strip():
                title = v.strip()
                break
        if not title:
            title = "Untitled story"

        key_takeaways: List[str] = []
        if isinstance(takeaways, list):
            key_takeaways = [x for x in takeaways if isinstance(x, str)][:10]

        query_parts = [title] + key_takeaways[:3]
        query = " | ".join([p.strip() for p in query_parts if p and p.strip()])

        summary = story_text.strip()[:1200] if isinstance(story_text, str) else ""

        # URL extraction: depends heavily on SDK response; keep empty unless clearly present
        urls: List[str] = []
        if latest_update is not None:
            for bucket in ("cluster_articles", "prompt_articles"):
                items = getattr(latest_update, bucket, None)
                if isinstance(items, list):
                    for it in items:
                        u = getattr(it, "url", None) or getattr(it, "link", None)
                        if isinstance(u, str) and u.startswith("http"):
                            urls.append(u)
        # dedup
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))][:10]

        support_count = int(n_articles) if isinstance(n_articles, (int, float)) else len(urls)

        topics.append(
            Topic(
                topic_id=f"stories-{safe_slug(title)}",
                mode="stories",
                title=title,
                query=query,
                summary=summary,
                key_takeaways=key_takeaways,
                source_urls=urls,
                support_count=support_count,
            )
        )

    return topics


# ============================================================
# NEWS -> Topic clustering (fallback when Stories forbidden)
# ============================================================
def validate_articles(raw_items: Sequence[Any]) -> List[ArticleDoc]:
    """Create ArticleDoc objects directly from SDK items."""
    docs: List[ArticleDoc] = []
    for item in raw_items:
        if isinstance(item, tuple) and len(item) == 2:
            candidate = item[1]
            if isinstance(candidate, (list, tuple)):
                for sub in candidate:
                    docs.extend(validate_articles([sub]))
                continue
            item = candidate
        doc = ArticleDoc.from_any(item)
        try:
            doc = ArticleDoc.model_validate(item, from_attributes=True)
        except Exception:
            # If an item is already a dict-like, allow it as input.
            doc = ArticleDoc.model_validate(item)
        # keep only usable docs
        if doc.best_title() or doc.best_summary():
            docs.append(doc)
    return docs


def topics_from_news_articles(
    articles: Sequence[ArticleDoc],
    max_topics: int = 12,
) -> List[Topic]:
    if not articles:
        return []

    corpus = [a.as_corpus_text() for a in articles]
    urls = [a.best_url() for a in articles]
    titles = [a.best_title() for a in articles]
    summaries = [a.best_summary() for a in articles]

    # Preferred: TF-IDF + KMeans (if available)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        k = min(max_topics, max(2, int(math.sqrt(len(corpus)))))
        vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        X = vectorizer.fit_transform(corpus)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        terms = vectorizer.get_feature_names_out()
        topics: List[Topic] = []

        for cluster_id in range(k):
            idxs = [i for i, lab in enumerate(labels) if lab == cluster_id]
            if not idxs:
                continue

            centroid = km.cluster_centers_[cluster_id]
            top_term_ids = centroid.argsort()[-8:][::-1]
            top_terms = [terms[j] for j in top_term_ids if j < len(terms)]
            topic_title = " / ".join(top_terms[:3]) if top_terms else f"Topic {cluster_id + 1}"

            rep_i = idxs[0]
            rep_title = titles[rep_i] or topic_title
            rep_summary = summaries[rep_i] or corpus[rep_i]

            # takeaways: representative headlines
            takeaways = [titles[i] for i in idxs if titles[i]][:10]
            cluster_urls = [urls[i] for i in idxs if urls[i]][:10]

            topics.append(
                Topic(
                    topic_id=f"news-{safe_slug(topic_title)}",
                    mode="news",
                    title=topic_title,
                    query=rep_title,
                    summary=rep_summary[:1200],
                    key_takeaways=takeaways,
                    source_urls=cluster_urls,
                    support_count=len(idxs),
                )
            )

        topics.sort(key=lambda t: t.support_count, reverse=True)
        return topics[:max_topics]

    except Exception:
        # Lightweight fallback: bucket by most common bigram per document
        buckets: List[str] = []
        bucket_counts: Dict[str, int] = {}

        for a in articles:
            toks = tokenize(a.as_corpus_text())
            bigrams = [" ".join([toks[i], toks[i + 1]]) for i in range(len(toks) - 1)]
            local: Dict[str, int] = {}
            for bg in bigrams:
                local[bg] = local.get(bg, 0) + 1
            bucket = max(local.items(), key=lambda x: x[1])[0] if local else (toks[0] if toks else "topic")
            buckets.append(bucket)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        top = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:max_topics]
        out: List[Topic] = []

        for bucket, _ in top:
            idxs = [i for i, b in enumerate(buckets) if b == bucket]
            rep_i = idxs[0]
            takeaways = [titles[i] for i in idxs if titles[i]][:10]
            cluster_urls = [urls[i] for i in idxs if urls[i]][:10]

            out.append(
                Topic(
                    topic_id=f"news-{safe_slug(bucket)}",
                    mode="news",
                    title=bucket,
                    query=titles[rep_i] or bucket,
                    summary=(summaries[rep_i] or corpus[rep_i])[:1200],
                    key_takeaways=takeaways,
                    source_urls=cluster_urls,
                    support_count=len(idxs),
                )
            )

        return out


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="AskNews → Topic Generator", layout="wide")
st.title("AskNews → Topic Generator")

with st.sidebar:
    st.header("Mode")
    mode = st.selectbox(
        "Topic source",
        options=["AUTO (Stories → News fallback)", "STORIES ONLY", "NEWS ONLY"],
        index=0,
    )

    st.header("LLM")
    llm_model = st.selectbox(
        "Modèle OpenRouter",
        options=[
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/openai/gpt-4o-mini",
            "mistralai/mixtral-8x7b-instruct",
        ],
        index=0,
    )
    llm_instructions = st.text_area(
        "Instructions pour ajuster les sujets (optionnel)",
        value="",
        height=120,
        help="Laisse vide pour conserver les sujets bruts.",
    )

    st.divider()
    st.header("Filters")
    query = st.text_input("Query (optional)", value="")
    method = st.selectbox("Search method", options=["kw", "nl", "both"], index=0)
    lookback_hours = st.slider("Lookback window (hours)", 6, 168, 24, 6)

    st.caption("These category labels may be SDK/endpoint dependent; signature filtering will drop unsupported params.")
    categories = st.multiselect(
        "Categories (optional)",
        options=["Business","Crime","Politics","Science","Sports","Technology","Military","Health","Entertainment"],
        default=[],
    )

    continents = st.multiselect(
        "Continents (optional)",
        options=["Europe","North America","South America","Africa","Asia","Oceania"],
        default=[],
    )

    languages = st.multiselect(
        "Languages (optional)",
        options=["en","fr","de","es","it","pt","nl","sv","no","da","pl","tr","ar","he","ru","uk","zh","ja","ko"],
        default=["en"],
    )

    limit = st.slider("Result limit", 5, 50, 15, 5)
    max_topics = st.slider("Max topics (news clustering)", 5, 25, 12, 1)
    topic_limit = st.slider("Nombre de sujets à retourner", 3, 25, 12, 1)

    show_debug = st.checkbox("Show diagnostics", value=True)
    run = st.button("Generate topics", type="primary")


# Validate API key early
try:
    _ = get_client()
except Exception as e:
    st.error(str(e))
    st.stop()


if not run:
    st.info(
        "Click **Generate topics**. If your API key cannot access Stories (403000), the app will automatically "
        "fallback to News-based topic clustering."
    )
    st.stop()


now_ts = now_utc_ts()
start_ts = now_ts - int(lookback_hours * 3600)

topics: List[Topic] = []
diagnostics: Dict[str, Any] = {}

wants_stories = mode in ("AUTO (Stories → News fallback)", "STORIES ONLY")
wants_news = mode in ("AUTO (Stories → News fallback)", "NEWS ONLY")

# ---------------------------
# 1) Try Stories (if enabled)
# ---------------------------
if wants_stories:
    stories_params: Dict[str, Any] = {
        "query": query or None,
        "method": method,
        # Some SDK versions accept obj_type as list; others as string. Signature filtering + server validation applies.
        "obj_type": ["story"],
        "categories": [c.lower() for c in categories] or None,
        "continent": continents[0] if len(continents) == 1 else None,
        "start_timestamp": start_ts,
        "end_timestamp": now_ts,
        "offset": 0,
        "limit": int(limit),
        "expand_updates": True,
        "max_updates": 2,
        "max_articles": 12,
        "reddit": 2,
        "sort_by": "coverage",
        "sort_type": "desc",
        "citation_method": "none",
    }
    stories_params = {k: v for k, v in stories_params.items() if v is not None}

    if show_debug:
        with st.expander("Diagnostics: stories params", expanded=False):
            st.json(stories_params)

    try:
        with st.spinner("Fetching AskNews Stories…"):
            stories_items = cached_stories(stories_params)
        topics = stories_to_topics(stories_items)

    except Exception as e:
        diagnostics["stories_error"] = extract_api_error_details(e)

        if is_invalid_permissions(e):
            st.warning(
                "Your API key does not have permission to access the Stories endpoint "
                "(403000: Invalid Permissions). Falling back to News."
            )
            if mode == "STORIES ONLY":
                st.error("Mode is STORIES ONLY, so the app cannot continue without Stories permission.")
                if show_debug:
                    st.json(diagnostics["stories_error"])
                st.stop()
        else:
            st.error("Stories call failed for a reason other than Invalid Permissions.")
            st.exception(e)
            if show_debug:
                st.json(diagnostics["stories_error"])
            st.stop()

# ---------------------------
# 2) News fallback (or News only)
# ---------------------------
if (not topics) and wants_news:
    news_params: Dict[str, Any] = {
        "query": query or None,
        "method": method,
        # Different SDK versions use different names; signature filtering will keep only supported keys.
        "n_articles": int(limit),
        "start_timestamp": start_ts,
        "end_timestamp": now_ts,
        "historical": bool(lookback_hours > 48),
        "return_type": "dicts",  # may be ignored/dropped by signature filtering
        "categories": categories or None,
        "continents": continents or None,
        "languages": languages or None,
        "offset": 0,
    }
    news_params = {k: v for k, v in news_params.items() if v is not None}

    if show_debug:
        with st.expander("Diagnostics: news params", expanded=False):
            st.json(news_params)

    try:
        with st.spinner("Fetching AskNews News…"):
            news_items = cached_news(news_params)

        if show_debug:
            with st.expander("Diagnostics: first 2 raw news items (object view)", expanded=False):
                # Show a safe minimal representation
                previews = []
                for it in news_items[:2]:
                    doc = ArticleDoc.from_any(it)
                    if doc.best_title() or doc.best_summary():
                        previews.append(doc.model_dump())
                    else:
                        previews.append({"repr": repr(it)})
                st.json(previews)

        articles = validate_articles(news_items)
        topics = topics_from_news_articles(articles, max_topics=int(max_topics))

    except Exception as e:
        diagnostics["news_error"] = extract_api_error_details(e)
        st.error("News call failed.")
        st.exception(e)
        if show_debug:
            st.json(diagnostics["news_error"])
        st.stop()

# ---------------------------
# Optional LLM refinement
# ---------------------------
topics = topics[: int(topic_limit)]

if llm_instructions.strip():
    if not OPENROUTER_API_KEY.strip():
        st.warning("OPENROUTER_API_KEY manquant : aucun affinage LLM effectué.")
    else:
        with st.spinner("Affinage des sujets via LLM…"):
            refined = refine_topics_with_llm(topics, llm_instructions, llm_model, int(topic_limit))
            if refined:
                topics = refined

# ---------------------------
# Output
# ---------------------------
st.subheader(f"Generated topics ({len(topics)})")

if not topics:
    st.warning("No topics generated. Try increasing the lookback window or loosening filters.")
    st.stop()

table_rows = [
    {
        "topic_id": t.topic_id,
        "mode": t.mode,
        "title": t.title,
        "support_count": t.support_count,
        "query_for_question_generation": t.query,
    }
    for t in topics
]
st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Topic details")

for t in topics:
    with st.expander(f"[{t.mode}] {t.title}", expanded=False):
        st.write("**Topic ID:**", t.topic_id)
        st.write("**Support count:**", t.support_count)
        st.write("**Query (for your next question generation stage):**")
        st.code(t.query)

        if t.key_takeaways:
            st.write("**Key takeaways / representative headlines:**")
            for kt in t.key_takeaways:
                st.write(f"- {kt}")

        if t.summary:
            st.write("**Summary:**")
            st.write(t.summary)

        if t.source_urls:
            st.write("**Source URLs (sample):**")
            for u in t.source_urls:
                st.write(f"- {u}")

topics_json = json.dumps([t.model_dump() for t in topics], ensure_ascii=False, indent=2)
st.download_button("Download topics.json", topics_json, file_name="topics.json", mime="application/json")

csv_cols = ["topic_id", "mode", "title", "support_count", "query", "summary"]
lines = [",".join(csv_cols)]


def esc(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace('"', '""')
    return f'"{s}"'


for t in topics:
    d = t.model_dump()
    lines.append(",".join(esc(d.get(c)) for c in csv_cols))

st.download_button("Download topics.csv", "\n".join(lines), file_name="topics.csv", mime="text/csv")

if show_debug and diagnostics:
    with st.expander("Diagnostics: errors encountered (if any)", expanded=False):
        st.json(diagnostics)
