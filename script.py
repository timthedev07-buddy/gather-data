import argparse
import json
import re
from typing import Any, Dict, Iterable, List, Optional


def _first_present(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_competition_name(value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    candidate = (value or default or "").strip().lower()
    if "amc 8" in candidate or "amc8" in candidate:
        return "amc8"
    if "amc 10" in candidate or "amc10" in candidate:
        return "amc10"
    if "amc 12" in candidate or "amc12" in candidate:
        return "amc12"
    if "aime" in candidate:
        return "aime"
    return candidate or None


def should_include_easy2hard_record(record: Dict[str, Any]) -> bool:
    combined = " ".join(str(v).lower() for v in record.values() if isinstance(v, str))
    return bool(re.search(r"\bamc\s*(8|10|12)\b", combined))


def normalize_record(record: Dict[str, Any], default_competition: Optional[str] = None) -> Optional[Dict[str, Any]]:
    problem = _first_present(record, ["problem", "question", "prompt", "statement", "input"])
    answer = _first_present(record, ["answer", "final_answer", "target", "solution", "label"])
    competition_raw = _first_present(record, ["competition", "source", "dataset", "contest", "exam"])
    topic = _first_present(record, ["topic", "subject", "category"])

    competition = normalize_competition_name(
        str(competition_raw) if competition_raw is not None else None,
        default=default_competition,
    )
    if problem is None or answer is None or competition is None:
        return None

    normalized: Dict[str, Any] = {
        "problem": str(problem),
        "answer": str(answer),
        "competition": competition,
    }
    if topic not in (None, ""):
        normalized["topic"] = str(topic)
    return normalized


def _iter_rows(dataset_obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(dataset_obj, dict):
        for split in dataset_obj.values():
            for row in split:
                yield row
        return
    for row in dataset_obj:
        yield row


def build_questions(output_path: str, write_empty_on_failure: bool = False) -> int:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("Please install the 'datasets' package: pip install datasets") from exc

    questions: List[Dict[str, Any]] = []
    errors: List[str] = []
    fetch_exceptions = (ConnectionError, OSError, RuntimeError, ValueError)

    try:
        aime = load_dataset("MathArena/aime")
        for row in _iter_rows(aime):
            normalized = normalize_record(row, default_competition="aime")
            if normalized:
                questions.append(normalized)
    except fetch_exceptions as exc:  # pragma: no cover - network/runtime dependent
        errors.append(f"MathArena/aime: {exc}")

    try:
        configs = get_dataset_config_names("furonghuang-lab/Easy2Hard-Bench")
        if not configs:
            configs = [None]
        for config in configs:
            dataset = load_dataset("furonghuang-lab/Easy2Hard-Bench", config)
            for row in _iter_rows(dataset):
                if not should_include_easy2hard_record(row):
                    continue
                normalized = normalize_record(row)
                if normalized and normalized["competition"] in {"amc8", "amc10", "amc12"}:
                    questions.append(normalized)
    except fetch_exceptions as exc:  # pragma: no cover - network/runtime dependent
        errors.append(f"furonghuang-lab/Easy2Hard-Bench: {exc}")

    if not questions and errors and not write_empty_on_failure:
        raise RuntimeError("Failed to fetch datasets: " + " | ".join(errors))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and combine AIME + AMC questions from HuggingFace datasets.")
    parser.add_argument(
        "--output",
        default="all_math_questions.json",
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--write-empty-on-failure",
        action="store_true",
        help="Write an empty JSON list if remote datasets are unavailable.",
    )
    args = parser.parse_args()
    total = build_questions(args.output, write_empty_on_failure=args.write_empty_on_failure)
    print(f"Wrote {total} questions to {args.output}")


if __name__ == "__main__":
    main()
