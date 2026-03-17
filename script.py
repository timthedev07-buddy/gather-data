import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import Request, urlopen


RECENT_WINDOW_YEARS = 5
RECENT_CONTEST_PREFIXES = ("AIME", "AMC 8", "AMC 10", "AMC 12")
OPEN_ENDED_HINTS = ("prove", "proof", "show that", "justify", "explain")


@dataclass
class DatasetSource:
    name: str
    url: str
    source_type: str


@dataclass
class UnifiedProblem:
    source: str
    contest: str
    year: int
    topic: str
    question_latex: str
    answer: str
    difficulty: str


DEFAULT_SOURCES: List[DatasetSource] = [
    DatasetSource(
        name="hf_competition_math",
        source_type="huggingface",
        url="https://datasets-server.huggingface.co/first-rows?dataset=EleutherAI%2Fhendrycks_math&config=competition_math&split=test",
    ),
    DatasetSource(
        name="aops_math_data",
        source_type="online",
        url="https://raw.githubusercontent.com/ArturUlfeldt/mathcontests-data/main/contests.json",
    ),
]


def _read_json(url: str, timeout: int = 30) -> Any:
    req = Request(url, headers={"User-Agent": "gather-data-script/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # nosec B310 - fixed URL list/user-provided URL
        return json.loads(resp.read().decode("utf-8"))


def _iter_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        return (r for r in payload if isinstance(r, dict))
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return (row.get("row", {}) for row in payload["rows"] if isinstance(row, dict))
        if isinstance(payload.get("data"), list):
            return (r for r in payload["data"] if isinstance(r, dict))
    return ()


def _extract_text(record: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _extract_year(record: Dict[str, Any]) -> Optional[int]:
    year = record.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    contest_date = record.get("date")
    if isinstance(contest_date, str):
        for token in contest_date.replace("/", "-").split("-"):
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def _is_recent_contest(contest: str, year: int, now_year: int) -> bool:
    if any(contest.upper().startswith(prefix) for prefix in (p.upper() for p in RECENT_CONTEST_PREFIXES)):
        return year >= now_year - (RECENT_WINDOW_YEARS - 1)
    return True


def _contains_latex(text: str) -> bool:
    return any(token in text for token in ("$", "\\(", "\\[", "\\frac", "\\sqrt"))


def _to_latex(question: str) -> str:
    if _contains_latex(question):
        return question
    escaped = question.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return f"\\text{{{escaped}}}"


def _is_open_ended(record: Dict[str, Any], question: str, answer: str) -> bool:
    lower_q = question.lower()
    if any(hint in lower_q for hint in OPEN_ENDED_HINTS):
        return True
    q_type = _extract_text(record, ("question_type", "type", "format")).lower()
    if q_type in {"proof", "open_ended", "essay", "free_response"}:
        return True
    return not bool(answer.strip())


def normalize_record(record: Dict[str, Any], source_name: str, now_year: Optional[int] = None) -> Optional[UnifiedProblem]:
    contest = _extract_text(record, ("contest", "exam", "source", "competition"), "unknown")
    year = _extract_year(record)
    topic = _extract_text(record, ("topic", "subject", "category", "domain"), "general")
    question = _extract_text(record, ("question_latex", "question", "problem", "prompt"))
    answer = _extract_text(record, ("answer", "final_answer", "solution"))
    difficulty = _extract_text(record, ("difficulty", "level"), "unknown")
    if year is None or not question:
        return None
    check_year = now_year if now_year is not None else datetime.now(timezone.utc).year
    if not _is_recent_contest(contest, year, check_year):
        return None
    if _is_open_ended(record, question, answer):
        return None
    return UnifiedProblem(
        source=source_name,
        contest=contest,
        year=year,
        topic=topic,
        question_latex=_to_latex(question),
        answer=answer,
        difficulty=difficulty,
    )


def gather_math_resources(
    sources: Optional[List[DatasetSource]] = None, payload_override: Optional[Dict[str, Any]] = None
) -> List[UnifiedProblem]:
    chosen_sources = sources if sources is not None else DEFAULT_SOURCES
    data = payload_override or {}
    unified: List[UnifiedProblem] = []
    for source in chosen_sources:
        payload = data.get(source.name)
        if payload is None:
            try:
                payload = _read_json(source.url)
            except Exception:
                continue
        for raw in _iter_records(payload):
            item = normalize_record(raw, source.name)
            if item is not None:
                unified.append(item)
    unified.sort(key=lambda x: (x.year, x.contest, x.topic))
    return unified


def save_unified_dataset(output_path: str, records: List[UnifiedProblem]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather AIME/AMC and related math contest data into a uniform format."
    )
    parser.add_argument("--output", default="data/unified_math_problems.jsonl")
    args = parser.parse_args()
    records = gather_math_resources()
    save_unified_dataset(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
