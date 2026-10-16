#!/usr/bin/env python
"""
Open-vocabulary support for Model 1.

Turns whatever the user typed (an object name like "headphones" or a task like
"something used to hear music") into a short list of object names that
YOLO-World is then asked to detect.

CLIP links a *use* to an object much better when the object is described by
what it is for, so every vocabulary entry carries a short affordance phrase:
the task is matched against "headphones, used to listen to music or audio",
not against the bare word "headphones".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# COCO-80 plus everyday objects an assistive robot is likely to be asked for,
# each with what it is used for.
OBJECT_USES: Dict[str, str] = {
    # COCO-80
    "person": "a human being",
    "bicycle": "ride and pedal to travel",
    "car": "drive and travel by road",
    "motorcycle": "ride fast on the road",
    "airplane": "fly and travel by air",
    "bus": "public transport for many passengers",
    "train": "travel by rail",
    "truck": "carry heavy cargo by road",
    "boat": "travel on water",
    "traffic light": "control road traffic",
    "fire hydrant": "supply water to fight fires",
    "stop sign": "tell vehicles to stop",
    "parking meter": "pay for parking",
    "bench": "sit on outdoors",
    "bird": "a flying animal",
    "cat": "a pet animal to cuddle",
    "dog": "a pet animal to walk and play with",
    "horse": "ride an animal for transport",
    "sheep": "a farm animal for wool",
    "cow": "a farm animal for milk",
    "elephant": "a large wild animal",
    "bear": "a large wild animal",
    "zebra": "a striped wild animal",
    "giraffe": "a tall wild animal",
    "backpack": "carry belongings on your back",
    "umbrella": "stay dry in the rain",
    "handbag": "carry personal things",
    "tie": "wear with a formal shirt",
    "suitcase": "pack clothes for travel",
    "frisbee": "throw and catch in play",
    "skis": "slide on snow",
    "snowboard": "ride down snowy slopes",
    "sports ball": "play a ball game",
    "kite": "fly in the wind",
    "baseball bat": "hit a ball",
    "baseball glove": "catch a ball",
    "skateboard": "ride and do tricks",
    "surfboard": "ride ocean waves",
    "tennis racket": "hit a tennis ball",
    "bottle": "hold and drink water or liquid",
    "wine glass": "serve and drink wine",
    "cup": "drink tea, coffee or water from",
    "fork": "eat food by picking it up",
    "knife": "cut food or spread butter",
    "spoon": "eat soup or stir",
    "bowl": "hold soup, cereal or food",
    "banana": "a fruit to eat",
    "apple": "a fruit to eat",
    "sandwich": "food to eat when hungry",
    "orange": "a fruit to eat or juice",
    "broccoli": "a vegetable to eat",
    "carrot": "a vegetable to eat",
    "hot dog": "food to eat when hungry",
    "pizza": "food to eat when hungry",
    "donut": "a sweet snack to eat",
    "cake": "a sweet dessert to eat",
    "chair": "sit on as a seat",
    "couch": "sit on or lie down and relax",
    "potted plant": "a plant to water and decorate",
    "bed": "sleep and rest",
    "dining table": "eat meals at",
    "toilet": "use the bathroom",
    "tv": "watch shows and movies",
    "laptop": "work, browse and type on a computer",
    "mouse": "point and click on a computer",
    "remote": "change the TV channel",
    "keyboard": "type on a computer",
    "cell phone": "call someone, text and use apps",
    "microwave": "heat food quickly",
    "oven": "bake and cook food",
    "toaster": "toast bread",
    "sink": "wash hands and dishes with water",
    "refrigerator": "keep food cold",
    "book": "read stories and learn",
    "clock": "tell the time",
    "vase": "hold and display flowers",
    "scissors": "cut paper or cloth",
    "teddy bear": "a soft toy to hug",
    "hair drier": "dry hair with hot air",
    "toothbrush": "brush and clean teeth",
    # Everyday / assistive extras
    "headphones": "listen to music or audio",
    "earphones": "listen to music or calls in your ears",
    "speaker": "play music out loud",
    "charger": "charge a phone battery",
    "power bank": "charge devices on the go",
    "cable": "connect or charge devices",
    "glasses": "see clearly and read",
    "sunglasses": "protect eyes from the sun",
    "wallet": "carry money and cards",
    "keys": "lock and unlock doors",
    "watch": "tell the time on your wrist",
    "pen": "write on paper",
    "pencil": "write or draw",
    "marker": "write or draw boldly",
    "notebook": "write notes in",
    "paper": "write or print on",
    "envelope": "post a letter",
    "box": "store or pack things",
    "parcel": "a delivered package to open",
    "bag": "carry things around",
    "shoe": "wear on your feet to walk",
    "slipper": "wear on your feet at home",
    "hat": "cover your head from the sun",
    "cap": "cover your head from the sun",
    "jacket": "keep warm in the cold",
    "shirt": "wear on your upper body",
    "towel": "dry your hands, face or body",
    "pillow": "rest your head while sleeping",
    "blanket": "keep warm in bed",
    "lamp": "give light to read",
    "fan": "cool down when hot",
    "tablet": "watch videos, read and browse",
    "monitor": "display a computer screen",
    "camera": "take photos",
    "water bottle": "carry and drink water",
    "mug": "drink hot tea or coffee",
    "plate": "serve and eat food on",
    "jar": "store food or sugar",
    "can": "hold a drink or food",
    "glass": "drink water or juice from",
    "kettle": "boil water for tea",
    "pan": "fry and cook food",
    "pot": "boil and cook food",
    "ladle": "serve soup",
    "spatula": "flip food while cooking",
    "cutting board": "chop vegetables on",
    "medicine": "take when sick",
    "pill bottle": "hold tablets and pills",
    "tissue box": "wipe your nose",
    "soap": "wash and clean hands",
    "toothpaste": "clean teeth with a brush",
    "comb": "comb and tidy hair",
    "brush": "brush hair or clean",
    "mirror": "see your reflection",
    "trash can": "throw away rubbish",
    "bucket": "carry water",
    "broom": "sweep the floor",
    "mop": "clean and wipe the floor",
    "watering can": "water plants",
    "flower": "decorate or give as a gift",
    "plant": "grow and water",
    "door": "enter or leave a room",
    "window": "let in light and fresh air",
    "light switch": "turn the light on or off",
    "screwdriver": "turn screws",
    "hammer": "hit nails",
    "wrench": "tighten nuts and bolts",
    "tape": "stick and seal things",
    "stapler": "staple papers together",
    "calculator": "do maths calculations",
    "ball": "throw and play",
    "toy": "play with",
    "guitar": "play music on strings",
    "walking stick": "support walking",
    "wheelchair": "move around when unable to walk",
    "face mask": "cover the nose and mouth",
    # Classroom / study
    "whiteboard": "write and teach on the wall with markers",
    "blackboard": "write and teach on with chalk",
    "chalk": "write on a blackboard",
    "duster": "erase and wipe the board clean",
    "eraser": "rub out pencil mistakes",
    "sharpener": "sharpen a pencil",
    "pencil sharpener": "sharpen a pencil",
    "ring binder": "keep notes and papers filed",
    "picture frame": "display a photo",
    "ruler": "measure and draw straight lines",
    "pencil case": "keep pens and pencils together",
    "geometry box": "hold a compass and protractor for maths",
    "highlighter": "highlight important text",
    "sticky notes": "write quick reminders",
    "file folder": "keep papers and documents organised",
    "clipboard": "hold paper while writing",
    "textbook": "study a school subject",
    "exam paper": "write answers in a test",
    "id card": "show identity and enter campus",
    "desk": "sit and study or write at",
    "student bench": "sit and study in a classroom",
    "podium": "stand and give a lecture",
    "projector": "show slides on a screen",
    "projector screen": "display projected slides",
    "presentation clicker": "change presentation slides",
    "microphone": "speak louder to an audience",
    "smart board": "teach with an interactive screen",
    "printer": "print documents on paper",
    "pen drive": "store and carry files",
    "laptop charger": "charge a laptop",
    "extension board": "plug in many devices",
    "router": "provide wifi internet",
    "bookshelf": "store and arrange books",
    "notice board": "pin up announcements",
    "poster": "display information on a wall",
    "dustbin": "throw away paper and rubbish",
    "lunch box": "carry food for lunch",
    "tiffin": "carry home-cooked food",
    "water jug": "pour water into glasses",
    # Room / surroundings
    "ceiling fan": "cool a room from the ceiling",
    "air conditioner": "cool the room air",
    "tube light": "light up a room",
    "curtain": "block light from a window",
    "cupboard": "store clothes and things",
    "drawer": "keep small things inside",
    "shelf": "keep things on",
    "table": "put things on or work at",
    "stool": "sit on without a backrest",
    "sofa": "sit on and relax",
    "mattress": "sleep on",
    "bedsheet": "cover a mattress",
    "hanger": "hang clothes",
    "clothes": "wear on the body",
    "socks": "wear on feet inside shoes",
    "sandals": "wear on feet outdoors",
    "helmet": "protect the head while riding",
    "earbuds": "listen to music wirelessly",
    "smartwatch": "track fitness and tell the time",
    "power socket": "plug in electric devices",
    "remote control": "control a fan, AC or TV",
    "first aid kit": "treat small injuries",
    "hand sanitizer": "clean and disinfect hands",
    "napkin": "wipe your mouth or hands",
    "doormat": "wipe shoes before entering",
}
OBJECT_VOCAB: List[str] = list(OBJECT_USES)

_FILLER_PREFIX = re.compile(
    r"^(please\s+)?(find|get|give me|bring( me)?|pick( up)?|show( me)?|where is|i want|i need)\s+",
    re.IGNORECASE,
)
_ARTICLE_PREFIX = re.compile(r"^(a|an|the|my|some)\s+", re.IGNORECASE)
# Words that mean the prompt describes a need, not an object name.
_TASK_WORDS = re.compile(
    r"^(to|for|something|anything|thing|object|item|stuff)\b|\b(used|use|to|for|with)\b",
    re.IGNORECASE,
)
PERSON_WORDS = re.compile(
    r"\b(person|people|man|woman|men|women|boy|girl|child|kid|human|someone|somebody|face)\b",
    re.IGNORECASE,
)


def mentions_person(task_text: str) -> bool:
    return bool(PERSON_WORDS.search(task_text))


def describe(label: str) -> str:
    """Affordance text for a class label, used for CLIP text matching."""
    use = OBJECT_USES.get(label.lower())
    return f"{label}, used to {use}" if use else label


def _clean_phrase(text: str) -> str:
    phrase = text.strip().strip(".?!").strip()
    phrase = _FILLER_PREFIX.sub("", phrase)
    phrase = _ARTICLE_PREFIX.sub("", phrase)
    return phrase.strip()


def _is_object_phrase(phrase: str) -> bool:
    """Short noun phrases ("headphones", "red cup") are object names; "to call someone" is not."""
    return 0 < len(phrase.split()) <= 3 and not _TASK_WORDS.search(phrase)


_vocab_emb_cache: dict = {}
HUB_WEIGHT = 1.0


@torch.no_grad()
def _vocab_embeddings(clip_model: Any, tokenizer: Any, device: str) -> tuple:
    """
    CLIP embeddings of every entry's affordance text plus its "hubness" bias,
    computed once and reused.

    Some entries ("toy", "book", "cake") are close to almost any sentence in
    CLIP space; their mean similarity to the other entries is subtracted so
    they stop crowding every list.
    """
    key = (id(clip_model), device)
    if key not in _vocab_emb_cache:
        texts = [describe(obj) for obj in OBJECT_VOCAB]
        embs = F.normalize(clip_model.encode_text(tokenizer(texts).to(device)).float(), dim=-1)
        pairwise = embs @ embs.T
        hub_bias = (pairwise.sum(dim=1) - 1.0) / (len(OBJECT_VOCAB) - 1)
        _vocab_emb_cache[key] = (embs, hub_bias)
    return _vocab_emb_cache[key]


@torch.no_grad()
def rank_vocabulary(task_text: str, clip_model: Any, tokenizer: Any, device: str) -> List[tuple]:
    """All vocabulary entries ranked by how well their use matches the task: [(name, sim), ...]."""
    vocab_emb, hub_bias = _vocab_embeddings(clip_model, tokenizer, device)
    task_emb = F.normalize(clip_model.encode_text(tokenizer([task_text]).to(device)).float(), dim=-1)
    sims = ((task_emb @ vocab_emb.T).squeeze(0) - HUB_WEIGHT * hub_bias).tolist()
    return sorted(zip(OBJECT_VOCAB, sims), key=lambda item: item[1], reverse=True)


def build_vocabulary(
    task_text: str,
    clip_model: Any,
    tokenizer: Any,
    device: str,
    top_k: int = 8,
) -> List[str]:
    """
    Build the YOLO-World class list for a user prompt.

    - A short object phrase ("headphones", "red cup") goes in first, as typed.
    - The top_k vocabulary entries whose use best matches the prompt are added.
      For task prompts they are the candidates ("hear music" -> headphones,
      earphones, speaker); for object phrases they let YOLO-World catch the
      object under a related name, and Model 2's image ranking picks the box.
    - "person" is left out unless the prompt is about a person, otherwise it is
      detected in nearly every frame and crowds out the real target.
    """
    phrase = _clean_phrase(task_text)
    allow_person = mentions_person(task_text)

    classes: List[str] = []
    if _is_object_phrase(phrase):
        classes.append(phrase.lower())
    limit = len(classes) + top_k
    for name, _ in rank_vocabulary(task_text, clip_model, tokenizer, device):
        if len(classes) >= limit:
            break
        if name == "person" and not allow_person:
            continue
        if name not in classes:
            classes.append(name)
    return classes
