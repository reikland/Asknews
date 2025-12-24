import argparse
import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

ASKNEWS_API_KEY = os.getenv("ASKNEWS_API_KEY", "ank_tIpMbXiY2OSUWCU1RvO9IJkFbqVRUMO5HmNg2AGSjz")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

ASKNEWS_TOPICS_URL = "https://api.asknews.app/v1/news/topics"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def fetch_topics(limit: int, language: Optional[str] = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit}
    if language:
        params["language"] = language

    url = f"{ASKNEWS_TOPICS_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {ASKNEWS_API_KEY}"}

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


def refine_topics_with_llm(topics: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate news topics from AskNews and optionally refine them with an OpenRouter-hosted LLM."
        )
    )
    parser.add_argument("count", type=int, help="Number of topics to retrieve from AskNews")
    parser.add_argument(
        "--language",
        help="Optional language code for topics (if supported by AskNews)",
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        help="OpenRouter model to use for topic refinement (e.g., openai/gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default="topics.csv",
        help="Path to the CSV file to generate (default: topics.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = fetch_topics(args.count, language=args.language)

    if args.llm_model:
        rows = refine_topics_with_llm(topics, args.llm_model)
        include_context = True
    else:
        rows = topics
        include_context = False

    write_csv(rows, args.output, include_context)
    print(f"Saved {len(rows)} topic rows to {args.output}")


if __name__ == "__main__":
    main()
