import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from script import (
    _first_present,
    _iter_rows,
    build_questions,
    normalize_competition_name,
    normalize_record,
    should_include_easy2hard_record,
)


def _raise_network_error(*args, **kwargs):
    raise OSError("network unavailable")


class ScriptTests(unittest.TestCase):
    def test_normalize_competition(self):
        self.assertEqual(normalize_competition_name("AMC 10A"), "amc10")
        self.assertEqual(normalize_competition_name("aime"), "aime")
        self.assertEqual(normalize_competition_name(None, default="AMC12"), "amc12")

    def test_easy2hard_filter(self):
        self.assertTrue(should_include_easy2hard_record({"source": "official amc 8 set"}))
        self.assertFalse(should_include_easy2hard_record({"source": "gsm8k"}))

    def test_normalize_record(self):
        row = {
            "question": "Solve $x^2=1$",
            "answer": "1, -1",
            "competition": "AMC 12",
            "topic": "algebra",
        }
        normalized = normalize_record(row)
        self.assertEqual(
            normalized,
            {
                "problem": "Solve $x^2=1$",
                "answer": "1, -1",
                "competition": "amc12",
                "topic": "algebra",
            },
        )

    def test_first_present(self):
        row = {"a": "", "b": None, "c": "value"}
        self.assertEqual(_first_present(row, ["a", "b", "c"]), "value")
        self.assertIsNone(_first_present(row, ["x", "y"]))

    def test_iter_rows(self):
        split_data = {"train": [{"problem": "p1"}], "test": [{"problem": "p2"}]}
        self.assertEqual(list(_iter_rows(split_data)), [{"problem": "p1"}, {"problem": "p2"}])
        self.assertEqual(list(_iter_rows([{"problem": "p3"}])), [{"problem": "p3"}])

    def test_build_questions_with_mocked_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.json")
            calls = []

            def get_dataset_config_names(name):
                calls.append(("configs", name))
                return ["main"]

            def load_dataset(name, config=None):
                calls.append(("load", name, config))
                if name == "MathArena/aime":
                    return {"train": [{"problem": "AIME $x$", "answer": "42"}]}
                return {
                    "train": [
                        {"question": "AMC 10 problem", "answer": "7", "source": "amc10"},
                        {"question": "Other problem", "answer": "1", "source": "gsm8k"},
                    ]
                }

            fake_datasets = SimpleNamespace(
                get_dataset_config_names=get_dataset_config_names,
                load_dataset=load_dataset,
            )

            with patch.dict("sys.modules", {"datasets": fake_datasets}):
                total = build_questions(output_path)

            self.assertEqual(total, 2)
            self.assertIn(("configs", "furonghuang-lab/Easy2Hard-Bench"), calls)
            self.assertIn(("load", "MathArena/aime", None), calls)
            self.assertIn(("load", "furonghuang-lab/Easy2Hard-Bench", "main"), calls)
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data[0]["competition"], "aime")
            self.assertEqual(data[1]["competition"], "amc10")

    def test_build_questions_raises_when_fetch_fails_and_empty_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.json")

            fake_datasets = SimpleNamespace(
                get_dataset_config_names=_raise_network_error,
                load_dataset=_raise_network_error,
            )

            with patch.dict("sys.modules", {"datasets": fake_datasets}):
                with self.assertRaisesRegex(RuntimeError, "No questions were fetched from HuggingFace datasets"):
                    build_questions(output_path)

    def test_build_questions_can_write_empty_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.json")

            fake_datasets = SimpleNamespace(
                get_dataset_config_names=_raise_network_error,
                load_dataset=_raise_network_error,
            )

            with patch.dict("sys.modules", {"datasets": fake_datasets}):
                total = build_questions(output_path, write_empty_on_failure=True)

            self.assertEqual(total, 0)
            with open(output_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])


if __name__ == "__main__":
    unittest.main()
