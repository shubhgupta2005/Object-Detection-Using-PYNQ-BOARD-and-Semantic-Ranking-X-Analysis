"""
Semantic Task Parsing Layer for Model-2.

Converts natural-language task text into an explicit object target and
associated keywords so downstream ranking can reason over intent directly.
"""

from __future__ import annotations

from typing import Dict, List


_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "of",
    "for",
    "with",
    "on",
    "in",
    "out",
    "from",
    "something",
    "object",
    "used",
    "use",
    "that",
    "this",
    "is",
    "are",
    "be",
    "get",
}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str) -> List[str]:
    cleaned = text
    for ch in ",.!?;:\"()[]{}-_/":
        cleaned = cleaned.replace(ch, " ")
    return [tok for tok in cleaned.split() if tok]


def _noun_fallback(text: str) -> Dict[str, List[str] | str]:
    tokens = _tokenize(text)
    nouns = [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 2]

    if not nouns and tokens:
        nouns = tokens

    if nouns:
        target = nouns[0]
        keywords = []
        for tok in nouns:
            if tok not in keywords:
                keywords.append(tok)
        return {"target_object": target, "keywords": keywords}

    return {"target_object": "object", "keywords": ["object"]}


def parse_task_to_object(task_text: str) -> dict:
    """
    Parse natural-language task text into explicit object intent.

    Returns
    -------
    dict with keys:
      - target_object: canonical object target
      - keywords: list of canonical and supporting keywords
    """
    text = _normalize_text(task_text)

    # Canonical 14-task rules for explicit task -> object mapping.
    rule_map = {
        "step on something": ("mat", ["mat", "doormat", "rug"]),
        "sit comfortably": ("chair", ["chair", "seat", "couch", "bench"]),
        "place flowers": ("vase", ["vase", "flower vase"]),
        "get potatoes out of fire": ("spoon", ["spoon", "tongs"]),
        "water plant": ("watering can", ["watering can", "can"]),
        "get lemon out of tea": ("spoon", ["spoon"]),
        "dig hole": ("shovel", ["shovel", "spade"]),
        "open bottle of beer": ("bottle opener", ["bottle opener", "opener"]),
        "open parcel": ("knife", ["knife", "cutter"]),
        "serve wine": ("wine glass", ["wine glass", "glass"]),
        "pour sugar": ("spoon", ["spoon"]),
        "smear butter": ("knife", ["knife", "butter knife"]),
        "extinguish fire": ("fire extinguisher", ["fire extinguisher", "extinguisher"]),
        "pound carpet": ("stick", ["stick", "bat"]),
    }

    # Synonyms and shorthand intents for broader natural-language handling.
    synonym_rules = [
        (("drink wine", "wine"), ("wine glass", ["wine glass", "glass"])),
        (("cut", "slice", "chop"), ("knife", ["knife", "cutter"])),
        (("sit", "seat"), ("chair", ["chair", "seat", "couch"])),
        (("serve",), ("wine glass", ["wine glass", "glass"])),
    ]

    if text in rule_map:
        target, keywords = rule_map[text]
        return {"target_object": target, "keywords": keywords}

    for triggers, parsed in synonym_rules:
        if any(trigger in text for trigger in triggers):
            target, keywords = parsed
            return {"target_object": target, "keywords": keywords}

    return _noun_fallback(text)
