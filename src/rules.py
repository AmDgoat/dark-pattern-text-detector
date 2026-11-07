"""High-precision regex rules that support the DistilBERT model.

Each rule: (category, confidence, compiled pattern). They catch very formulaic
wording ("Only 2 left in stock", "27 people are looking at this") with near
certainty and also cover rare categories the model has little data for.
"""
import re

_R = [
    # --- Scarcity -----------------------------------------------------
    ("Scarcity", 0.98, r"\b(?:only|just)\s+(?:a\s+few|\d+|one|two|three|four|five)\s+(?:[\w()]+\s+){0,3}?(?:left|remaining|available|in stock)\b"),
    ("Scarcity", 0.98, r"\b(?:there are|there is)\s+only\s+\d+\b"),
    ("Scarcity", 0.97, r"\b(?:low|running low)\s+(?:in\s+)?stock\b"),
    ("Scarcity", 0.97, r"\bonly\s+limited\s+(?:items?|stock|quantit(?:y|ies))\s+left\b"),
    ("Scarcity", 0.97, r"\b(?:less|fewer) than\s+\d+\s+(?:items?\s+|units?\s+)?(?:left|available|remaining|in stock)\b"),
    ("Scarcity", 0.97, r"\b\d+\s+(?:items?\s+|units?\s+)?(?:left|remaining)\s+in stock\b"),
    ("Scarcity", 0.95, r"\b(?:low stock|almost (?:gone|sold out)|selling fast|limited stock|limited quantit(?:y|ies)|hurry,? (?:only )?(?:a )?few)\b"),
    # --- Urgency ------------------------------------------------------
    ("Urgency", 0.96, r"\b(?:offer|sale|deal|discount|promo(?:tion)?|price)s?\s+ends?\s+(?:in|on|at|today|tonight|soon|tomorrow|midnight)\b"),
    ("Urgency", 0.95, r"\bends?\s+in\s+\d"),
    ("Urgency", 0.95, r"\blimited[- ]time\b"),
    ("Urgency", 0.95, r"\b(?:today only|last chance|flash sale|ending soon|act now|don'?t miss out)\b"),
    ("Urgency", 0.95, r"\b(?:ends?|expires?|left|remaining)\b.{0,25}\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    ("Urgency", 0.90, r"\bhurry[!,.]"),
    # --- Social proof -------------------------------------------------
    ("Social Proof", 0.97, r"\b\d+\s+(?:other\s+)?(?:people|persons|customers|shoppers|users|buyers|visitors)\s+(?:are\s+|is\s+)?(?:currently\s+|right now\s+)?(?:viewing|looking|watching|have (?:bought|purchased)|bought|purchased|added)\b"),
    ("Social Proof", 0.95, r"\b(?:just|recently)\s+(?:bought|purchased|ordered)\b"),
    ("Social Proof", 0.95, r"\b(?:bought|purchased|ordered)\b.{0,40}\b(?:\d+\s+(?:seconds?|minutes?|hours?|days?)\s+ago|just now)\b"),
    ("Social Proof", 0.93, r"\b\d+\+?\s+(?:sold|bought|purchased)\s+(?:in the last|today|this (?:week|hour))\b"),
    # --- Misdirection (confirmshaming) --------------------------------
    ("Misdirection", 0.96, r"^\W*no,?\s*thanks?,?\s*i\s+(?:don'?t|do not|hate|prefer|would rather|like paying|am not)\b"),
    ("Misdirection", 0.95, r"\bi\s+(?:don'?t|do not)\s+(?:want|like|need)\s+(?:to\s+)?(?:save|saving|discounts?|deals?|money|benefits?)\b"),
    ("Misdirection", 0.95, r"\bno,?\s+i\s+(?:prefer|like|enjoy)\s+(?:paying|full price|missing)\b"),
    # --- Other (sneaking / obstruction / forced action) ----------------
    ("Other", 0.85, r"\bauto[- ]?renew(?:s|al|ed|ing)?\b"),
    ("Other", 0.85, r"\b(?:automatically|will be)\s+(?:enrolled|charged|billed|renewed)\b"),
    ("Other", 0.85, r"\bi\s+(?:would like to |want to )?(?:agree|consent)\s+to\s+receive\s+(?:marketing|promotional|emails?)\b"),
]
RULES = [(c, p, re.compile(rx, re.I)) for c, p, rx in _R]


def match_rules(text):
    """Return (category, confidence) of the best matching rule, or None."""
    best = None
    for cat, conf, rx in RULES:
        if rx.search(text) and (best is None or conf > best[1]):
            best = (cat, conf)
    return best
