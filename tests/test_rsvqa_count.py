import importlib.util
import unittest
from pathlib import Path

path = (
    Path(__file__).resolve().parents[1]
    / "internvl_chat"
    / "eval"
    / "domain_specific"
    / "rs_vqa"
    / "score.py"
)
spec = importlib.util.spec_from_file_location("rsvqa_score", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RsvqaCountTest(unittest.TestCase):
    def test_a_count_of_five_is_not_a_count_of_fifty(self):
        self.assertFalse(mod.is_correct_count(5, 50))

    def test_counts_inside_one_published_bucket_still_match(self):
        self.assertTrue(mod.is_correct_count(5, 9))
        self.assertTrue(mod.is_correct_count(11, 90))
        self.assertTrue(mod.is_correct_count(0, 0))
        self.assertTrue(mod.is_correct_count(101, 500))
        self.assertTrue(mod.is_correct_count(1001, 2000))


if __name__ == "__main__":
    unittest.main()
