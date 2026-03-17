import unittest

from script import DatasetSource, gather_math_resources, normalize_record


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
        rows = gather_math_resources(sources=sources, payload_override=payloads)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.topic, "counting")
        self.assertEqual(row.answer, "10")
        self.assertIn("question_latex", row.__dict__)


if __name__ == "__main__":
    unittest.main()
