"""
test_runner_offline.py
=======================
Runs all unit tests using a mock CLIP backend (no network / GPU required).
This lets you verify all model logic and utility functions offline.
When deployed with real open_clip installed, use test_task_aware_selector.py directly.
"""

import json
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

# ── Inject mock torch before importing the main module ──────────────────────
# We build a minimal torch-compatible namespace using only numpy.

class _Tensor:
    """Minimal torch.Tensor stand-in backed by numpy."""

    def __init__(self, data):
        if isinstance(data, _Tensor):
            self._d = data._d.copy()
        else:
            self._d = np.array(data, dtype=np.float32)

    # shape / dtype
    @property
    def shape(self):    return self._d.shape
    @property
    def dtype(self):    return self._d.dtype
    def size(self, dim=None):
        return self._d.shape[dim] if dim is not None else self._d.shape

    # indexing
    def __getitem__(self, idx):  return _Tensor(self._d[idx])
    def __setitem__(self, idx, v):
        self._d[idx] = v._d if isinstance(v, _Tensor) else v

    # arithmetic
    def __add__(self, o):  return _Tensor(self._d + (o._d if isinstance(o,_Tensor) else o))
    def __mul__(self, o):  return _Tensor(self._d * (o._d if isinstance(o,_Tensor) else o))
    def __sub__(self, o):  return _Tensor(self._d - (o._d if isinstance(o,_Tensor) else o))
    def __truediv__(self, o): return _Tensor(self._d / (o._d if isinstance(o,_Tensor) else o))
    def __matmul__(self, o):  return _Tensor(self._d @ (o._d if isinstance(o,_Tensor) else o))
    def __ge__(self, o):  return (self._d >= (o._d if isinstance(o,_Tensor) else o)).all()
    def __le__(self, o):  return (self._d <= (o._d if isinstance(o,_Tensor) else o)).all()
    def __neg__(self):    return _Tensor(-self._d)
    def all(self):        return bool(self._d.all())
    def item(self):       return float(self._d.flat[0])
    def tolist(self):     return self._d.tolist()
    def argmax(self, dim=None): return _Tensor(np.argmax(self._d, axis=dim))
    def sum(self, dim=None):
        if dim is None: return _Tensor(np.sum(self._d))
        return _Tensor(np.sum(self._d, axis=dim))
    def squeeze(self, dim=None): return _Tensor(np.squeeze(self._d, axis=dim))
    def expand(self, *shape):    return _Tensor(np.broadcast_to(self._d, shape))
    def clone(self):             return _Tensor(self._d.copy())
    def cpu(self):               return self
    def to(self, *a, **kw):      return self
    def float(self):             return self
    @property
    def device(self):            return "cpu"
    def T(self):                 return _Tensor(self._d.T)
    @property
    def T(self):                 return _Tensor(self._d.T)

    def __repr__(self):
        return f"Tensor({self._d})"


def _randn(*shape):
    return _Tensor(np.random.randn(*shape).astype(np.float32))

def _zeros(*shape, **kw):
    # shape may be passed as individual ints or as a tuple
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return _Tensor(np.zeros(shape, dtype=np.float32))

def _tensor(data, dtype=None, device=None):
    return _Tensor(data)


# Mock torch.nn
class _Linear:
    def __init__(self, in_f, out_f):
        self.w = _Tensor(np.random.randn(out_f, in_f).astype(np.float32) * 0.01)
        self.b = _Tensor(np.zeros(out_f, dtype=np.float32))
    def __call__(self, x):
        return _Tensor(x._d @ self.w._d.T + self.b._d)

class _ReLU:
    def __call__(self, x): return _Tensor(np.maximum(x._d, 0))

class _Dropout:
    def __init__(self, p=0.1): pass
    def __call__(self, x):     return x

class _Sequential:
    def __init__(self, *layers): self.layers = layers
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x

class _Module:
    def eval(self): return self
    def to(self, *a, **kw): return self
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)

class _MockNN:
    Linear     = _Linear
    ReLU       = _ReLU
    Dropout    = _Dropout
    Sequential = _Sequential
    Module     = _Module

class _MockF:
    @staticmethod
    def normalize(x, dim=-1):
        norm = np.linalg.norm(x._d, axis=dim, keepdims=True) + 1e-8
        return _Tensor(x._d / norm)

    @staticmethod
    def softmax(x, dim=-1):
        d = x._d
        d = d - d.max(axis=dim, keepdims=True)
        e = np.exp(d)
        return _Tensor(e / e.sum(axis=dim, keepdims=True))

# Build mock torch module
_torch = types.ModuleType("torch")
_torch.Tensor  = _Tensor
_torch.randn   = _randn
_torch.zeros   = _zeros
_torch.tensor  = _tensor
_torch.no_grad = lambda: __import__("contextlib").nullcontext()
_torch.cuda    = types.SimpleNamespace(is_available=lambda: False)
_torch.float32 = np.float32
_torch.nn      = _MockNN()
_torch.nn.functional = _MockF
_torch.nn.Module     = _Module

_f_mod = types.ModuleType("torch.nn.functional")
_f_mod.normalize = _MockF.normalize
_f_mod.softmax   = _MockF.softmax
sys.modules["torch"] = _torch
sys.modules["torch.nn"] = types.ModuleType("torch.nn")
sys.modules["torch.nn"].functional = _f_mod
sys.modules["torch.nn.functional"] = _f_mod

# Mock open_clip (returns random normalised embeddings)
_open_clip = types.ModuleType("open_clip")

class _FakeClipModel:
    def encode_text(self, tokens):
        n = tokens._d.shape[0] if hasattr(tokens, "_d") else len(tokens)
        return _Tensor(np.random.randn(n, 512).astype(np.float32))
    def eval(self): return self
    def to(self, *a, **kw): return self

def _fake_create(*a, **kw):
    return _FakeClipModel(), None, None

def _fake_tokenizer(texts):
    n = len(texts) if isinstance(texts, list) else 1
    return _Tensor(np.zeros((n, 77), dtype=np.float32))

_open_clip.create_model_and_transforms = _fake_create
_open_clip.get_tokenizer = lambda *a, **kw: _fake_tokenizer
sys.modules["open_clip"] = _open_clip

# ── Now import the module under test ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import importlib
import task_aware_selector as m
# Patch torch references inside the module to our mocks
m.torch = _torch
m.F     = _MockF

from task_aware_selector import (
    TASKS, NUM_TASKS, TIE_THRESHOLD, CLIP_DIM,
    HierarchicalTaskClassifier,
    load_detections,
    _filter_detections,
    compute_similarity,
    rank_objects,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rand_emb(n=1, dim=CLIP_DIM):
    t = _Tensor(np.random.randn(n, dim).astype(np.float32))
    return _MockF.normalize(t, dim=-1)

def _make_det(name="chair", conf=0.9, x=10, y=10, w=50, h=50):
    return {"class_name": name, "confidence": conf,
            "x": x, "y": y, "width": w, "height": h}


# ── Unit Tests ───────────────────────────────────────────────────────────────

class TestLoadDetections(unittest.TestCase):

    def test_loads_valid_file(self):
        data = {"detections": [_make_det("chair"), _make_det("table")]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f); fname = f.name
        try:
            dets = load_detections(fname)
            self.assertEqual(len(dets), 2)
            self.assertEqual(dets[0]["class_name"], "chair")
        finally:
            os.unlink(fname)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_detections("/nonexistent/path.json")

    def test_empty_detections_list(self):
        data = {"detections": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f); fname = f.name
        try:
            self.assertEqual(load_detections(fname), [])
        finally:
            os.unlink(fname)


class TestFilterDetections(unittest.TestCase):

    def test_passes_valid(self):
        self.assertEqual(len(_filter_detections([_make_det()])), 1)

    def test_removes_zero_confidence(self):
        self.assertEqual(_filter_detections([_make_det(conf=0.0)]), [])

    def test_removes_negative_confidence(self):
        self.assertEqual(_filter_detections([_make_det(conf=-0.5)]), [])

    def test_removes_negative_coords(self):
        self.assertEqual(_filter_detections([_make_det(x=-1)]), [])

    def test_removes_zero_area(self):
        self.assertEqual(_filter_detections([_make_det(w=0, h=0)]), [])

    def test_removes_empty_class_name(self):
        self.assertEqual(_filter_detections([_make_det(name="")]), [])

    def test_mixed_batch(self):
        dets = [
            _make_det("chair", conf=0.9),
            _make_det("ghost", conf=-0.1),
            _make_det("thing", conf=0.5, w=0),
            _make_det("table", conf=0.7),
        ]
        out = _filter_detections(dets)
        self.assertEqual(len(out), 2)
        names = {d["class_name"] for d in out}
        self.assertIn("chair", names)
        self.assertIn("table", names)


class TestHierarchicalClassifier(unittest.TestCase):

    def setUp(self):
        self.model = HierarchicalTaskClassifier(input_dim=CLIP_DIM)

    def test_output_shape(self):
        x = _rand_emb(4)
        out = self.model(x)
        self.assertEqual(out.shape, (4, NUM_TASKS))

    def test_probabilities_sum_to_one(self):
        x = _rand_emb(8)
        out = self.model(x)
        sums = out.sum(dim=-1)
        for i in range(8):
            s = float(out._d[i].sum())
            self.assertAlmostEqual(s, 1.0, places=4)

    def test_all_probs_non_negative(self):
        x = _rand_emb(4)
        out = self.model(x)
        self.assertTrue((out._d >= 0).all())

    def test_group_sizes(self):
        n_a = len(HierarchicalTaskClassifier.GROUP_A_IDX)
        n_b = len(HierarchicalTaskClassifier.GROUP_B_IDX)
        self.assertEqual(n_a + n_b, NUM_TASKS)

    def test_single_sample(self):
        x = _rand_emb(1)
        out = self.model(x)
        self.assertEqual(out.shape, (1, NUM_TASKS))
        self.assertAlmostEqual(float(out._d[0].sum()), 1.0, places=4)


class TestComputeSimilarity(unittest.TestCase):

    def test_shape(self):
        task = _rand_emb(1)
        objs = _rand_emb(5)
        scores, cos_sim = compute_similarity(task, objs, [0.9]*5)
        self.assertEqual(scores.shape, (5,))
        self.assertEqual(cos_sim.shape, (5,))

    def test_identical_embeddings(self):
        emb = _rand_emb(1)
        _, cos_sim = compute_similarity(emb, emb, [1.0])
        self.assertAlmostEqual(float(cos_sim._d[0]), 1.0, places=4)

    def test_confidence_scaling(self):
        task = _rand_emb(1)
        # Stack the same embedding 3 times → cos_sim ≈ 1
        d = np.tile(task._d, (3, 1)).astype(np.float32)
        objs = _Tensor(d)
        confs = [0.5, 0.8, 1.0]
        scores, _ = compute_similarity(task, objs, confs)
        for s, c in zip(scores._d.tolist(), confs):
            expected = 0.7 + (0.3 * c)
            self.assertAlmostEqual(s, expected, places=3)

    def test_scores_bounded(self):
        task = _rand_emb(1)
        objs = _rand_emb(10)
        scores, cos_sim = compute_similarity(task, objs, [0.9]*10)
        self.assertTrue((scores._d >= -1).all())
        self.assertTrue((scores._d <= 1.01).all())


class TestRankObjects(unittest.TestCase):

    def _inputs(self, n=4):
        dets = [_make_det(f"obj{i}", conf=round(0.9-i*0.1, 2)) for i in range(n)]
        scores  = _Tensor(np.random.rand(n).astype(np.float32))
        cos_sim = _Tensor(np.random.rand(n).astype(np.float32))
        return dets, scores, cos_sim

    def test_count(self):
        dets, scores, cos_sim = self._inputs(4)
        self.assertEqual(len(rank_objects(dets, scores, cos_sim)), 4)

    def test_descending_order(self):
        dets, scores, cos_sim = self._inputs(5)
        ranked = rank_objects(dets, scores, cos_sim)
        for i in range(len(ranked)-1):
            self.assertGreaterEqual(ranked[i]["score"], ranked[i+1]["score"])

    def test_enriched_fields(self):
        dets, scores, cos_sim = self._inputs(2)
        ranked = rank_objects(dets, scores, cos_sim)
        for obj in ranked:
            self.assertIn("score",   obj)
            self.assertIn("cos_sim", obj)

    def test_single(self):
        dets    = [_make_det("chair")]
        ranked  = rank_objects(dets, _Tensor([0.7]), _Tensor([0.8]))
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["class_name"], "chair")


class TestTaskConstants(unittest.TestCase):
    def test_count(self):  self.assertEqual(len(TASKS), 14)
    def test_unique(self): self.assertEqual(len(TASKS), len(set(TASKS)))


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestLoadDetections,
        TestFilterDetections,
        TestHierarchicalClassifier,
        TestComputeSimilarity,
        TestRankObjects,
        TestTaskConstants,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
