import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from script import (
    DEFAULT_SOURCES,
    DatasetSource,
    build_sources,
    gather_arcmath_recent,
    gather_math_resources,
    normalize_record,
    save_datasets_by_competition,
)


class ScriptTests(unittest.TestCase):
    def test_normalize_filters_proof_and_old_amc(self):
        self.assertIsNone(
            normalize_record(
                {
                    "contest": "AMC 10A",
                    "year": 2018,
                    "topic": "algebra",
                    "question": "Compute x",
                    "answer": "5",
                },
                source_name="x",
                now_year=2026,
            )
        )
        self.assertIsNone(
            normalize_record(
                {
                    "contest": "AIME I",
                    "year": 2025,
                    "topic": "geometry",
                    "question": "Prove that triangle ABC is isosceles.",
                    "answer": "See proof.",
                },
                source_name="x",
                now_year=2026,
            )
        )

    def test_normalize_adds_latex_wrapper(self):
        normalized = normalize_record(
            {
                "contest": "AMC 8",
                "year": 2026,
                "topic": "number theory",
                "question": "What is 2+2?",
                "answer": "4",
                "difficulty": "easy",
            },
            source_name="x",
            now_year=2026,
        )
        assert normalized is not None
        self.assertEqual(normalized.question_latex, r"\text{What is 2+2?}")

    def test_gather_math_resources_uses_uniform_schema(self):
        sources = [DatasetSource(name="mock", url="https://example.com", source_type="online")]
        payloads = {
            "mock": {
                "rows": [
                    {
                        "row": {
                            "contest": "AMC 12B",
                            "year": 2025,
                            "topic": "counting",
                            "question": r"What is $\binom{5}{2}$?",
                            "answer": "10",
                            "difficulty": "medium",
                        }
                    }
                ]
            }
        }
        rows = gather_math_resources(sources=sources, payload_override=payloads, include_arcmath=False)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.topic, "counting")
        self.assertEqual(row.answer, "10")
        self.assertIn("question_latex", row.__dict__)

    def test_build_sources_adds_extra_online_sources(self):
        sources = build_sources(["https://example.com/a.json", "https://example.com/b.json"])
        self.assertEqual(len(sources), len(DEFAULT_SOURCES) + 2)
        self.assertEqual(sources[-1].name, "online_source_2")

    def test_save_datasets_by_competition_creates_split_files(self):
        rows = [
            normalize_record(
                {
                    "contest": "AMC 8",
                    "year": 2026,
                    "topic": "number theory",
                    "question": "What is 2+2?",
                    "answer": "4",
                },
                source_name="x",
                now_year=2026,
            ),
            normalize_record(
                {
                    "contest": "AIME I",
                    "year": 2026,
                    "topic": "algebra",
                    "question": "Find x if x=3",
                    "answer": "3",
                },
                source_name="x",
                now_year=2026,
            ),
        ]
        rows = [r for r in rows if r is not None]
        with TemporaryDirectory() as tmp:
            save_datasets_by_competition(tmp, rows)
            amc_file = Path(tmp) / "amc_8.jsonl"
            aime_file = Path(tmp) / "aime_i.jsonl"
            self.assertTrue(amc_file.exists())
            self.assertTrue(aime_file.exists())

    @patch("script._read_json")
    @patch("script._arcmath_recent_files")
    def test_gather_arcmath_recent_transforms_problem_set(self, mock_recent_files, mock_read_json):
        # Input contest code follows source naming ("AMC8"), while output is normalized ("AMC 8").
        mock_recent_files.return_value = [
            {
                "contest": "AMC8",
                "year": 2026,
                "exam": None,
                "download_url": "https://example.com/amc8_2026.json",
            }
        ]
        mock_read_json.return_value = {
                "problemSet": {"contest": "AMC8", "year": 2026, "exam": None},
                "problems": [{"number": 1, "answer": "A", "sourceUrl": "https://example.com/p1"}],
            }
        rows = gather_arcmath_recent(now_year=2026)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].contest, "AMC 8")
        self.assertEqual(rows[0].difficulty, "easy")
        self.assertEqual(rows[0].answer, "A")
        self.assertIn("Problem 1", rows[0].question_latex)


if __name__ == "__main__":
    unittest.main()
