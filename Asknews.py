import csv
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

ASKNEWS_API_KEY = os.getenv("ASKNEWS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

ASKNEWS_TOPICS_URL = "https://api.asknews.app/v1/news/topics"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def fetch_topics(
    limit: int, language: Optional[str] = None, api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    resolved_api_key = api_key or ASKNEWS_API_KEY
    if not resolved_api_key:
        raise ValueError("Missing AskNews API key")

    params: Dict[str, Any] = {"limit": limit}
    if language:
        params["language"] = language

    url = f"{ASKNEWS_TOPICS_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {resolved_api_key}"}

    payload = _http_request(url, method="GET", headers=headers)
    if isinstance(payload, list):
        raw_topics = payload
    elif isinstance(payload, dict):
        if "topics" in payload:
            raw_topics = payload["topics"]
        elif "data" in payload and isinstance(payload["data"], dict) and "topics" in payload["data"]:
            raw_topics = payload["data"]["topics"]
        else:
            raise ValueError("Unexpected AskNews response format: missing 'topics' list")
    else:
        raise ValueError("Unexpected AskNews response type")

    return _normalize_topics(raw_topics, limit)


def _normalize_topics(raw_topics: Iterable[Any], limit: int) -> List[Dict[str, str]]:
    normalized = []
    for idx, item in enumerate(raw_topics):
        if idx >= limit:
            break
        if isinstance(item, str):
            topic = item.strip()
            summary = ""
        elif isinstance(item, dict):
            topic = _first_value(item, ["topic", "title", "name", "label", "headline"]) or ""
            summary = _first_value(item, ["summary", "description", "snippet", "text", "details"]) or ""
            if not topic:
                topic = summary or f"Topic {idx + 1}"
        else:
            topic = str(item)
            summary = ""
        normalized.append({"topic": topic, "summary": summary})
    if not normalized:
        raise ValueError("AskNews returned an empty topics list")
    return normalized


def _first_value(source: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def refine_topics_with_llm(
    topics: List[Dict[str, str]], model: str, api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    resolved_api_key = api_key or OPENROUTER_API_KEY
    if not resolved_api_key:
        raise ValueError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    prompt = (
        "You receive a list of news topics gathered from AskNews. For each topic, "
        "propose a short, crisp refined_topic (max 12 words) and an optional one-sentence context. "
        "Return a JSON array with objects containing 'topic', 'refined_topic', and 'context'. "
        "Keep the same order and length as the input list."
    )

    user_topics = [
        {"topic": item["topic"], "summary": item.get("summary", "")}
        for item in topics
    ]

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user_topics, ensure_ascii=False)},
        ],
        "temperature": 0.3,
    }

    completion = _http_request(OPENROUTER_CHAT_URL, method="POST", headers=headers, data=json.dumps(body))
    content = _extract_chat_content(completion)

    try:
        refined = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response could not be parsed as JSON") from exc

    if not isinstance(refined, list):
        raise ValueError("LLM response JSON is not a list")

    cleaned: List[Dict[str, str]] = []
    for original, item in zip(topics, refined):
        topic = original.get("topic", "")
        refined_topic = ""
        context = ""
        if isinstance(item, dict):
            refined_topic = _first_value(item, ["refined_topic", "title", "topic"]) or topic
            context = _first_value(item, ["context", "summary", "description", "note"]) or ""
        else:
            refined_topic = str(item)
        cleaned.append({"topic": topic, "refined_topic": refined_topic, "context": context})

    return cleaned


def _extract_chat_content(payload: Dict[str, Any]) -> str:
    if "choices" in payload and payload["choices"]:
        message = payload["choices"][0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("Unexpected LLM response format: missing content")


def _http_request(url: str, method: str, headers: Optional[Dict[str, str]] = None, data: Optional[str] = None) -> Any:
    request_headers = headers or {}
    encoded_data = None
    if data is not None:
        encoded_data = data.encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=encoded_data, headers=request_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            content = response.read().decode(charset)
            return json.loads(content)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else exc.reason
        raise RuntimeError(f"HTTP error {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error reaching {url}: {exc.reason}") from exc


def write_csv(rows: List[Dict[str, str]], output_file: str, include_context: bool) -> None:
    fieldnames = ["topic"]
    sample = rows[0] if rows else {}
    if "refined_topic" in sample:
        fieldnames.append("refined_topic")
    if include_context:
        fieldnames.append("context")
    if "summary" in sample:
        fieldnames.append("summary")

    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_csv_content(rows: List[Dict[str, str]], include_context: bool) -> str:
    fieldnames = ["topic"]
    sample = rows[0] if rows else {}
    if "refined_topic" in sample:
        fieldnames.append("refined_topic")
    if include_context:
        fieldnames.append("context")
    if "summary" in sample:
        fieldnames.append("summary")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def render_app() -> None:
    st.set_page_config(page_title="AskNews Topic Explorer", page_icon="📰")

    st.title("📰 AskNews Topic Explorer")
    st.write(
        "Générez rapidement des sujets d'actualité avec AskNews, puis affinez-les "
        "avec un modèle hébergé sur OpenRouter. Fournissez vos clés API pour utiliser les services."
    )

    with st.sidebar:
        st.header("Clés API")
        asknews_key = st.text_input(
            "AskNews API key", value=ASKNEWS_API_KEY or "", type="password"
        )
        openrouter_key = st.text_input(
            "OpenRouter API key (pour l'affinage)",
            value=OPENROUTER_API_KEY or "",
            type="password",
        )
        st.caption(
            "Les clés sont utilisées uniquement pendant cette session Streamlit et ne sont pas stockées."
        )

    st.subheader("Paramètres de récupération")
    count = st.number_input("Nombre de sujets", min_value=1, max_value=50, value=10, step=1)
    language = st.text_input("Langue (facultatif, ex: en, fr, es)")

    st.subheader("Affinage optionnel par LLM")
    llm_model = st.text_input(
        "Modèle OpenRouter (ex: openai/gpt-4o-mini)",
        help="Laisser vide pour ne pas affiner les sujets",
    )

    if st.button("Générer les sujets", type="primary"):
        try:
            with st.spinner("Récupération des sujets AskNews..."):
                topics = fetch_topics(
                    int(count), language=language or None, api_key=asknews_key or None
                )

            rows = topics
            include_context = False

            if llm_model.strip():
                with st.spinner("Affinage via le modèle OpenRouter..."):
                    rows = refine_topics_with_llm(
                        topics, llm_model.strip(), api_key=openrouter_key or None
                    )
                include_context = True

            st.success(f"{len(rows)} sujets prêts.")
            st.dataframe(rows, use_container_width=True)

            csv_content = build_csv_content(rows, include_context)
            st.download_button(
                "Télécharger en CSV",
                data=csv_content.encode("utf-8"),
                file_name="topics.csv",
                mime="text/csv",
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(str(exc))


if __name__ == "__main__":
    render_app()
