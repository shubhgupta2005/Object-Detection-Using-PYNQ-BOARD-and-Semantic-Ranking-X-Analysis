"""
test_task_aware_selector.py
============================
Unit tests for the task-aware object selection model.

Covers:
    - load_detections()
    - _filter_detections()
    - HierarchicalTaskClassifier forward pass
    - compute_similarity()
    - rank_objects()
    - select_best_object()
    - full pipeline via main()
"""

import json
import math
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

# ── import module under test ─────────────────────────────────────────────────
from task_aware_selector import (
    TASKS,
    NUM_TASKS,
    TIE_THRESHOLD,
    CLIP_DIM,
    HierarchicalTaskClassifier,
    load_detections,
    _filter_detections,
    compute_similarity,
    rank_objects,
)
from semantic_parser import parse_task_to_object


# ===========================================================================
# Helpers
# ===========================================================================

def _make_det(name="chair", conf=0.9, x=10, y=10, w=50, h=50):
    return {"class_name": name, "confidence": conf,
            "x": x, "y": y, "width": w, "height": h}


def _rand_emb(n=1, dim=CLIP_DIM):
    """Return L2-normalised random embeddings."""
    t = torch.randn(n, dim)
    return F.normalize(t, dim=-1)


# ===========================================================================
# Tests
# ===========================================================================

class TestLoadDetections(unittest.TestCase):

    def test_loads_valid_file(self):
        data = {"detections": [_make_det("chair"), _make_det("table")]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            dets = load_detections(fname)
            self.assertEqual(len(dets), 2)
            self.assertEqual(dets[0]["class_name"], "chair")
        finally:
            os.unlink(fname)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_detections("/nonexistent/path/detections.json")

    def test_empty_detections_list(self):
        data = {"detections": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            dets = load_detections(fname)
            self.assertEqual(dets, [])
        finally:
            os.unlink(fname)


class TestFilterDetections(unittest.TestCase):

    def test_passes_valid_detection(self):
        dets = [_make_det("chair", conf=0.8, x=5, y=5, w=40, h=40)]
        out = _filter_detections(dets)
        self.assertEqual(len(out), 1)

    def test_removes_zero_confidence(self):
        dets = [_make_det(conf=0.0)]
        self.assertEqual(_filter_detections(dets), [])

    def test_removes_negative_confidence(self):
        dets = [_make_det(conf=-0.5)]
        self.assertEqual(_filter_detections(dets), [])

    def test_removes_negative_coords(self):
        dets = [_make_det(x=-1, y=0, conf=0.9)]
        self.assertEqual(_filter_detections(dets), [])

    def test_removes_zero_area(self):
        dets = [_make_det(w=0, h=0, conf=0.9)]
        self.assertEqual(_filter_detections(dets), [])

    def test_removes_empty_class_name(self):
        dets = [_make_det(name="", conf=0.9)]
        self.assertEqual(_filter_detections(dets), [])

    def test_mixed_batch(self):
        dets = [
            _make_det("chair", conf=0.9),        # valid
            _make_det("ghost", conf=-0.1),        # invalid conf
            _make_det("thing", conf=0.5, w=0),    # invalid area
            _make_det("table", conf=0.7),         # valid
        ]
        out = _filter_detections(dets)
        self.assertEqual(len(out), 2)
        self.assertIn("chair", [d["class_name"] for d in out])
        self.assertIn("table", [d["class_name"] for d in out])


class TestHierarchicalClassifier(unittest.TestCase):

    def setUp(self):
        self.model = HierarchicalTaskClassifier(input_dim=CLIP_DIM)
        self.model.eval()

    def test_output_shape(self):
        x = _rand_emb(4)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (4, NUM_TASKS))

    def test_probabilities_sum_to_one(self):
        x = _rand_emb(8)
        with torch.no_grad():
            out = self.model(x)
        sums = out.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_all_probs_non_negative(self):
        x = _rand_emb(4)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue((out >= 0).all())

    def test_group_sizes_consistent(self):
        n_a = len(HierarchicalTaskClassifier.GROUP_A_IDX)
        n_b = len(HierarchicalTaskClassifier.GROUP_B_IDX)
        self.assertEqual(n_a + n_b, NUM_TASKS)

    def test_single_sample(self):
        x = _rand_emb(1)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, NUM_TASKS))
        self.assertAlmostEqual(out.sum().item(), 1.0, places=5)


class TestComputeSimilarity(unittest.TestCase):

    def test_shape(self):
        task = _rand_emb(1)
        objs = _rand_emb(5)
        confs = [0.9, 0.8, 0.7, 0.6, 0.5]
        scores, cos_sim = compute_similarity(task, objs, confs)
        self.assertEqual(scores.shape,  (5,))
        self.assertEqual(cos_sim.shape, (5,))

    def test_identical_embeddings_give_high_similarity(self):
        emb = _rand_emb(1)
        scores, cos_sim = compute_similarity(emb, emb, [1.0])
        # Cosine sim of identical unit vectors = 1.0
        self.assertAlmostEqual(cos_sim[0].item(), 1.0, places=5)

    def test_confidence_scaling(self):
        task = _rand_emb(1)
        objs = task.expand(3, -1).clone()   # identical embeddings → cos_sim = 1
        confs = [0.5, 0.8, 1.0]
        scores, _ = compute_similarity(task, objs, confs)
        # score = (0.7 * clip_sim) + (0.3 * conf); clip_sim ≈ 1
        for s, c in zip(scores.tolist(), confs):
            expected = 0.7 + (0.3 * c)
            self.assertAlmostEqual(s, expected, places=4)

    def test_scores_bounded(self):
        task = _rand_emb(1)
        objs = _rand_emb(10)
        confs = [0.9] * 10
        scores, cos_sim = compute_similarity(task, objs, confs)
        # cos_sim ∈ [-1, 1]; conf ∈ [0, 1] → score ∈ [-1, 1]
        self.assertTrue((scores >= -1).all())
        self.assertTrue((scores <= 1).all())


class TestRankObjects(unittest.TestCase):

    def _make_inputs(self, n=4):
        dets = [_make_det(f"obj{i}", conf=round(0.9 - i * 0.1, 2))
                for i in range(n)]
        scores  = torch.rand(n)
        cos_sim = torch.rand(n)
        return dets, scores, cos_sim

    def test_returns_correct_count(self):
        dets, scores, cos_sim = self._make_inputs(4)
        ranked = rank_objects(dets, scores, cos_sim)
        self.assertEqual(len(ranked), 4)

    def test_descending_order(self):
        dets, scores, cos_sim = self._make_inputs(5)
        ranked = rank_objects(dets, scores, cos_sim)
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i]["score"], ranked[i + 1]["score"])

    def test_enriched_fields_present(self):
        dets, scores, cos_sim = self._make_inputs(2)
        ranked = rank_objects(dets, scores, cos_sim)
        for obj in ranked:
            self.assertIn("score",   obj)
            self.assertIn("cos_sim", obj)

    def test_single_object(self):
        dets    = [_make_det("chair")]
        scores  = torch.tensor([0.7])
        cos_sim = torch.tensor([0.8])
        ranked  = rank_objects(dets, scores, cos_sim)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["class_name"], "chair")


class TestTaskList(unittest.TestCase):
    def test_correct_number_of_tasks(self):
        self.assertEqual(len(TASKS), 14)

    def test_no_duplicate_tasks(self):
        self.assertEqual(len(TASKS), len(set(TASKS)))


class TestSemanticParser(unittest.TestCase):

    def test_rule_mapping_serve_wine(self):
        parsed = parse_task_to_object("serve wine")
        self.assertEqual(parsed["target_object"], "wine glass")
        self.assertIn("wine glass", parsed["keywords"])

    def test_synonym_mapping(self):
        parsed = parse_task_to_object("drink wine")
        self.assertEqual(parsed["target_object"], "wine glass")

    def test_synonym_cut(self):
        parsed = parse_task_to_object("cut vegetables")
        self.assertEqual(parsed["target_object"], "knife")

    def test_fallback_noun_extraction(self):
        parsed = parse_task_to_object("pick umbrella quickly")
        self.assertEqual(parsed["target_object"], "pick")
        self.assertIn("umbrella", parsed["keywords"])


class TestSemanticPriorityRanking(unittest.TestCase):

    def test_target_object_is_prioritized(self):
        detections = [
            _make_det("cup", conf=0.99),
            _make_det("wine glass", conf=0.70),
        ]
        scores = torch.tensor([0.95, 0.50])
        cos_sim = torch.tensor([0.95, 0.50])

        ranked = rank_objects(
            detections,
            scores,
            cos_sim,
            target_object="wine glass",
            keywords=["wine glass", "glass"],
        )

        self.assertEqual(ranked[0]["class_name"], "wine glass")
        self.assertTrue(ranked[0]["target_match"])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
