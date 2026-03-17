import argparse
import ipaddress
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RECENT_WINDOW_YEARS = 5
RECENT_CONTEST_PREFIXES = ("AIME", "AMC 8", "AMC 10", "AMC 12")
OPEN_ENDED_HINTS = ("prove", "proof", "show that", "justify", "explain")


@dataclass
class DatasetSource:
    """Remote JSON source descriptor.

    name: unique identifier used in logs/overrides.
    url: absolute endpoint used to fetch JSON payload.
    source_type: provenance label (for example 'huggingface' or 'online').
    """

    name: str
    url: str
    source_type: str


@dataclass
class UnifiedProblem:
    """Normalized math problem record.

    difficulty is one of: easy, medium, hard, very_hard, unknown.
    """

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
]
ARCMATH_BASE_URL = (
    "https://raw.githubusercontent.com/xiaoming1348/Arcmath/3420f12ea41380da7798150687aba7c372b4b37b/"
    "packages/db/data/aops-imports"
)


def _read_json(url: str, timeout: int = 30) -> Any:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError(f"Unsupported URL: {url}")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Localhost URLs are not allowed.")
    ip = None
    try:
        ip = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast):
        raise ValueError("Private or local IP addresses are not allowed.")
    req = Request(url, headers={"User-Agent": "gather-data-script/1.0"})
    with urlopen(req, timeout=timeout) as resp:
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


def _infer_topic(question: str) -> str:
    q = question.lower()
    topic_rules = [
        ("geometry", ("triangle", "circle", "angle", "polygon", "parallel", "perpendicular", "area", "perimeter")),
        ("number theory", ("integer", "prime", "divisible", "gcd", "lcm", "mod", "remainder", "factor")),
        ("combinatorics", ("arrange", "choose", "permutation", "combination", "ways", "probability", "count")),
        ("algebra", ("equation", "polynomial", "function", "solve", "variable", "system")),
    ]
    for topic, keywords in topic_rules:
        if any(k in q for k in keywords):
            return topic
    return "general"


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


def _build_contest_label(contest: str, exam: Optional[str]) -> str:
    exam_part = f" {exam}" if exam else ""
    if contest == "AMC8":
        return "AMC 8"
    if contest == "AMC10":
        return f"AMC 10{exam_part}".strip()
    if contest == "AMC12":
        return f"AMC 12{exam_part}".strip()
    if contest == "AIME":
        return f"AIME{exam_part}".strip()
    return contest


def _difficulty_for_contest(contest: str) -> str:
    return {
        "AMC8": "easy",
        "AMC10": "medium",
        "AMC12": "hard",
        "AIME": "very_hard",
    }.get(contest, "unknown")


def _arcmath_recent_files(now_year: Optional[int] = None) -> List[Dict[str, Any]]:
    check_year = now_year if now_year is not None else datetime.now(timezone.utc).year
    chosen = []
    for year in range(check_year - (RECENT_WINDOW_YEARS - 1), check_year + 1):
        for contest, exams in (
            ("AMC8", [None]),
            ("AMC10", ["A", "B"]),
            ("AMC12", ["A", "B"]),
            ("AIME", ["I", "II"]),
        ):
            for exam in exams:
                label = _build_contest_label(contest, exam)
                if not _is_recent_contest(label, year, check_year):
                    continue
                name = f"{contest}_{year}{f'_{exam}' if exam else ''}.json"
                chosen.append(
                    {
                        "contest": contest,
                        "year": year,
                        "exam": exam,
                        "download_url": f"{ARCMATH_BASE_URL}/{name}",
                    }
                )
    return chosen


def gather_arcmath_recent(now_year: Optional[int] = None) -> List[UnifiedProblem]:
    problems: List[UnifiedProblem] = []
    for file_meta in _arcmath_recent_files(now_year=now_year):
        url = file_meta.get("download_url")
        if not isinstance(url, str) or not url:
            continue
        try:
            payload = _read_json(url)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for problem in payload.get("problems", []):
            if not isinstance(problem, dict):
                continue
            answer = str(problem.get("answer", "")).strip()
            if not answer:
                continue
            contest = _build_contest_label(file_meta["contest"], file_meta.get("exam"))
            number = problem.get("number", "?")
            statement = str(problem.get("statement", "")).strip()
            question = statement or f"Problem {number} from {contest} {file_meta['year']} (statement unavailable in source)."
            topic = _infer_topic(statement)
            normalized = normalize_record(
                {
                    "contest": contest,
                    "year": file_meta["year"],
                    "topic": topic,
                    "question_latex": question,
                    "answer": answer,
                    "difficulty": _difficulty_for_contest(file_meta["contest"]),
                },
                source_name="online_arcmath",
                now_year=now_year,
            )
            if normalized is not None:
                problems.append(normalized)
    return problems


def gather_math_resources(
    sources: Optional[List[DatasetSource]] = None,
    payload_override: Optional[Dict[str, Any]] = None,
    include_arcmath: bool = True,
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
    if include_arcmath:
        unified.extend(gather_arcmath_recent())
    unified.sort(key=lambda x: (x.year, x.contest, x.topic))
    return unified


def build_sources(extra_online_urls: Optional[List[str]] = None) -> List[DatasetSource]:
    sources = list(DEFAULT_SOURCES)
    for idx, url in enumerate(extra_online_urls or []):
        sources.append(DatasetSource(name=f"online_source_{idx + 1}", url=url, source_type="online"))
    return sources


def save_unified_dataset(output_path: str, records: List[UnifiedProblem]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _competition_filename(contest: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", contest.lower()).strip("_")
    return f"{slug or 'unknown'}.jsonl"


def save_datasets_by_competition(output_dir: str, records: List[UnifiedProblem]) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[UnifiedProblem]] = {}
    for row in records:
        grouped.setdefault(row.contest, []).append(row)
    for contest, items in grouped.items():
        path = directory / _competition_filename(contest)
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather AIME/AMC and related math contest data into a uniform format."
    )
    parser.add_argument("--output", default="data/unified_math_problems.jsonl")
    parser.add_argument("--split-output-dir", default="data/by_competition")
    parser.add_argument(
        "--online-source-url",
        action="append",
        default=[],
        help="Additional online JSON endpoint(s) to ingest besides Hugging Face datasets.",
    )
    args = parser.parse_args()
    records = gather_math_resources(sources=build_sources(args.online_source_url))
    save_unified_dataset(args.output, records)
    save_datasets_by_competition(args.split_output_dir, records)
    print(f"Wrote {len(records)} records to {args.output} and split files in {args.split_output_dir}")


if __name__ == "__main__":
    main()
