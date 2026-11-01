"""Shared helpers: BIO -> entity spans, prompt format, and scoring."""

import json
import re

COARSE = {
    "PER": "Person",
    "ORG": "Organisation",
    "LOC": "Location",
    "NRM": "Norm (statute, regulation, contract)",
    "REG": "Regulation (administrative rule, decree)",
    "RS": "Case law (court decision)",
    "LIT": "Legal literature",
}

SYSTEM = (
    "Extract the legal named entities from the German sentence. "
    "Reply with only a JSON array of objects with keys \"text\" and \"type\". "
    "Allowed types: " + ", ".join(COARSE) + ". "
    "Use the exact text from the sentence. Reply [] if there are none."
)


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    entities, cur_tokens, cur_type = [], [], None

    def flush():
        nonlocal cur_tokens, cur_type
        if cur_tokens:
            entities.append({"text": " ".join(cur_tokens), "type": cur_type})
        cur_tokens, cur_type = [], None

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            flush()
        elif tag.startswith("B-") or tag[2:] != cur_type:
            flush()
            cur_tokens, cur_type = [tok], tag[2:]
        else:
            cur_tokens.append(tok)
    flush()
    return entities


def messages(sentence: str, answer: list[dict] | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": sentence},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return msgs


def parse_prediction(raw: str) -> list[dict] | None:
    """Return a list of {text,type} dicts, or None if the output is not valid."""
    raw = raw.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if not (isinstance(item, dict) and isinstance(item.get("text"), str)
                and item.get("type") in COARSE):
            return None
        out.append({"text": item["text"], "type": item["type"]})
    return out


def entity_key(e: dict) -> tuple[str, str]:
    return (e["text"], e["type"])
