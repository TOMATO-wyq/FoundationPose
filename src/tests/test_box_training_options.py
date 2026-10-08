"""Unlabelled images must be opt-in negatives; new training has no early stop."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import box_train_seg as training


class TrainingOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "labelme_json").mkdir()
        rows = [dict(id=f"frame_{i:03d}", file=f"frame_{i:03d}.png", group="capture",
                     source_frame=i, split="train") for i in range(60)]
        self.rows = rows
        (self.root / "manifest.json").write_text(json.dumps(rows))
        for row in rows[:40]:
            (self.root / "labelme_json" / f"{row['id']}.json").write_text(json.dumps({"flags": {"reviewed": True}, "shapes": [{}]}))
        root_patch = patch.object(training, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_new_training_defaults_to_150_without_early_stopping(self):
        args = training.args_parser().parse_args([])
        self.assertEqual((args.epochs, args.patience), (150, 0))

    def test_unused_images_are_not_implicitly_negative(self):
        rows, _, report = training.split_rows(training.args_parser().parse_args([]))
        self.assertFalse(any(r.get("confirmed_negative") for r in rows))
        self.assertEqual(report["missing_json"], 20)
        self.assertTrue(all(int(r["id"].split("_")[1]) < 40 for r in rows))

    def test_default_confirmed_list_only_applies_to_new_training(self):
        path = self.root / "confirmed.txt"
        path.write_text("frame_059\n")
        with patch.object(training, "DEFAULT_NEGATIVES", path):
            args = training.resolve_negative_ids(training.args_parser().parse_args([]))
            self.assertEqual(args.confirmed_negative_ids, path)
            args = training.resolve_negative_ids(training.args_parser().parse_args(["--resume", "last.pt"]))
            self.assertIsNone(args.confirmed_negative_ids)
            args = training.resolve_negative_ids(training.args_parser().parse_args(["--no-confirmed-negatives"]))
            self.assertIsNone(args.confirmed_negative_ids)

    def test_explicit_negatives_participate_in_split_and_gap(self):
        path = self.root / "confirmed.txt"
        path.write_text("\n".join(r["id"] for r in self.rows[40:]))
        args = training.args_parser().parse_args(["--confirmed-negative-ids", str(path)])
        rows, _, report = training.split_rows(args)
        self.assertEqual(len(rows), 40)  # ten frames on BOTH sides of the mixed validation block
        self.assertEqual(report["counts"], {"train": 28, "val": 12})
        self.assertEqual(report["confirmed_negative_counts"], {"val": 4, "train": 6})
        validation = [r for r in rows if r['split'] == 'val']
        self.assertTrue(any(r.get('confirmed_negative') for r in validation))
        self.assertTrue(any(not r.get('confirmed_negative') for r in validation))
        self.assertEqual(len(list((self.root / "labelme_json").glob("*.json"))), 40)
        self.assertEqual(json.loads((self.root / "manifest.json").read_text()), self.rows)

    def test_existing_annotation_cannot_be_overwritten_as_negative(self):
        path = self.root / "confirmed.txt"
        path.write_text("frame_000\n")
        args = training.args_parser().parse_args(["--confirmed-negative-ids", str(path)])
        with self.assertRaisesRegex(ValueError, "已有 JSON"):
            training.split_rows(args)

    def test_unknown_id_rejected(self):
        path = self.root / "confirmed.txt"
        path.write_text("not_in_manifest\n")
        args = training.args_parser().parse_args(["--confirmed-negative-ids", str(path)])
        with self.assertRaisesRegex(ValueError, "manifest"):
            training.split_rows(args)


if __name__ == "__main__":
    unittest.main()
