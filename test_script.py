import unittest

from script import normalize_competition_name, normalize_record, should_include_easy2hard_record


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


if __name__ == "__main__":
    unittest.main()
