#!/usr/bin/env python
"""
Task-aware Model 2 ranker.

Consumes YOLO detections + task text and selects the best candidate index.
Designed to match the second-model interface expected by TaskAwareObjectIdentifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class CandidateScore:
    index: int
    class_name: str
    confidence: float
    bbox: List[float]
    area: float
    score: float


class TaskAwareRankerModel(nn.Module):
    """Simple deterministic ranking model for task-conditioned object selection."""

    def __init__(self) -> None:
        super().__init__()

    def _task_class_weights(self, task: str) -> Dict[str, float]:
        """Map task text to class preference weights."""
        text = (task or "").strip().lower()

        # Includes typo-tolerant token for the user-provided phrase "comfotably".
        if "sit" in text or "seat" in text or "comfortable" in text or "comfortably" in text or "comfotably" in text:
            return {
                "couch": 0.70,
                "chair": 0.65,
                "bench": 0.55,
                "bed": 0.50,
                "dining table": 0.20,
            }

        # Generic default when task is unknown.
        return {}

    def _bbox_area(self, bbox: Sequence[float]) -> float:
        if len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def rank_detections(self, detections: List[Dict], task: str) -> List[CandidateScore]:
        """Compute ranking scores for each detection candidate."""
        class_weights = self._task_class_weights(task)

        areas = [self._bbox_area(det.get("bbox", [])) for det in detections]
        max_area = max(areas) if areas else 1.0
        if max_area <= 0:
            max_area = 1.0

        ranked: List[CandidateScore] = []
        for idx, det in enumerate(detections):
            class_name = str(det.get("class_name", "unknown"))
            confidence = float(det.get("confidence", 0.0))
            bbox = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
            area = self._bbox_area(bbox)

            class_bonus = class_weights.get(class_name, 0.0)
            area_bonus = 0.10 * (area / max_area)

            # Final score combines detector confidence + task relevance + size prior.
            score = confidence + class_bonus + area_bonus

            ranked.append(
                CandidateScore(
                    index=idx,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    area=area,
                    score=score,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def forward(
        self,
        input_image: torch.Tensor | None = None,
        task: str = "",
        detections: List[Dict] | None = None,
        class_ids: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        boxes_xyxy: torch.Tensor | None = None,
    ) -> Dict:
        """
        Task-aware inference entry point.

        Returns dict with selected_index and ranked candidates.
        """
        candidates = detections or []
        if not candidates:
            return {
                "selected_index": -1,
                "ranked_candidates": [],
                "reason": "no_detections",
            }

        ranked = self.rank_detections(candidates, task=task)
        selected = ranked[0]

        return {
            "selected_index": int(selected.index),
            "selected_object": {
                "class_name": selected.class_name,
                "confidence": selected.confidence,
                "bbox": selected.bbox,
            },
            "ranked_candidates": [
                {
                    "index": int(item.index),
                    "class_name": item.class_name,
                    "confidence": float(item.confidence),
                    "bbox": item.bbox,
                    "score": float(round(item.score, 6)),
                }
                for item in ranked
            ],
        }
