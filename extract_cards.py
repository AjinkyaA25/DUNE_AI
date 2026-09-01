"""
Card image extraction script for Dune Imperium: Uprising.
Uses Anthropic vision API to extract structured card data from PNG images.
Outputs:
  src/data/uprising_cards.json
  src/game/cards/card_definitions.py
"""
import anthropic
import base64
import json
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGES_DIR = Path(r"C:\Users\argon\Ajinkya\Consules videos\Images\cards\Uprising")
OUTPUT_JSON = Path(r"C:\Users\argon\Ajinkya\Dune_AI\src\data\uprising_cards.json")
OUTPUT_PY   = Path(r"C:\Users\argon\Ajinkya\Dune_AI\src\game\cards\card_definitions.py")
FAQ_PATH    = Path(r"C:\Users\argon\Ajinkya\Dune_AI\Dune Imperium - Comprehensive Rules FAQ.txt")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are extracting structured data from a Dune Imperium: Uprising card image.
Return ONLY valid JSON â€” no markdown fences, no explanation, no extra text.

Card layout (top to bottom, left to right):
  TOP-LEFT:     faction tag (the coloured faction banner label)
  TOP-CENTER:   card name
  TOP-RIGHT:    purchase cost (integer; number of persuasion required to buy)
  BELOW COST:   acquire effect â€” text or icon that triggers immediately on purchase (null if absent)
  MIDDLE:       agent icons â€” coloured diamond/circle symbols granting access to board spaces:
                  Yellow   = Maker (desert/spice spaces)
                  Purple   = City (Arrakeen, Carthag, etc.)
                  Green    = Landsraad
                  Dark Purple = Bene Gesserit
                  Black    = Emperor
                  Red      = Spacing Guild
                  Blue     = Fremen
                (a card may have 0, 1, or multiple icons)
  MIDDLE-BOTTOM BOX: agent effect â€” what happens when played as an agent turn (null if none)
  BOTTOM STRIP: reveal effect â€” icons showing:
                  a numbered pentagon/shield = persuasion
                  a sword/dagger = swords for combat (count the icons)
                  spice/solari/water resource icons

Return exactly this JSON structure, nothing else:
{
  "name": "card name as written on card",
  "faction_tag": "faction shown top-left, or null if none",
  "cost": 0,
  "acquire_effect": null,
  "agent_icons": ["Maker", "City"],
  "agent_effect": "text description of agent effect, or null",
  "reveal_effect": {
    "persuasion": 0,
    "swords": 0,
    "resources": []
  }
}

For resources use strings like "+1 spice", "+1 solari", "+1 water", "+1 troop",
"draw 1", "+1 influence[Emperor]", "deploy 1 to conflict", "place spy", "+1 VP".
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_card(image_path: Path) -> dict:
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract all card data and return as JSON only.",
                },
            ],
        }],
    )
    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def to_snake_case(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def extract_faq_snippet(faq_text: str, card_name: str) -> str | None:
    """Return the FAQ paragraph(s) nearest to a card name mention."""
    idx = faq_text.lower().find(card_name.lower())
    if idx == -1:
        return None
    # Walk back to paragraph start
    start = faq_text.rfind("\n\n", 0, idx)
    start = start + 2 if start != -1 else max(0, idx - 100)
    # Walk forward to paragraph end (up to 600 chars)
    end = faq_text.find("\n\n", idx + len(card_name))
    end = end if (end != -1 and end - start < 800) else min(len(faq_text), start + 700)
    return faq_text[start:end].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

image_files = sorted(IMAGES_DIR.glob("*.PNG")) + sorted(IMAGES_DIR.glob("*.png"))
# Deduplicate (case-insensitive), keep one per stem
seen: set[str] = set()
deduped: list[Path] = []
for f in image_files:
    key = f.name.lower()
    if key not in seen:
        seen.add(key)
        deduped.append(f)
image_files = sorted(deduped, key=lambda x: x.name.lower())

print(f"Found {len(image_files)} card images\n")

cards: list[dict] = []
failed: list[dict] = []
BATCH_SIZE = 10

for i, img_path in enumerate(image_files):
    print(f"[{i+1:02d}/{len(image_files)}] {img_path.name} ...", end=" ", flush=True)
    try:
        card_data = extract_card(img_path)
        card_data["_source_file"] = img_path.name
        cards.append(card_data)
        print(f"OK  â†’ {card_data.get('name', '?')}  cost={card_data.get('cost', '?')}")
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        failed.append({"file": img_path.name, "error": f"JSONDecodeError: {e}"})
    except Exception as e:
        print(f"ERROR: {e}")
        failed.append({"file": img_path.name, "error": str(e)})

    if (i + 1) % BATCH_SIZE == 0 and (i + 1) < len(image_files):
        print("  -- batch pause 1s --")
        time.sleep(1)

print(f"\nExtracted {len(cards)} cards, {len(failed)} failures.\n")

# ---------------------------------------------------------------------------
# FAQ cross-reference
# ---------------------------------------------------------------------------
faq_text = ""
if FAQ_PATH.exists():
    faq_text = FAQ_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"FAQ loaded ({len(faq_text):,} chars)")
else:
    print(f"FAQ not found at {FAQ_PATH}")

faq_annotated = 0
for card in cards:
    name = card.get("name", "")
    if name and faq_text:
        snippet = extract_faq_snippet(faq_text, name)
        if snippet:
            card["faq_notes"] = snippet
            faq_annotated += 1

# Check FAQ card names not in our set
faq_card_names_not_found: list[str] = []
our_names_lower = {c.get("name", "").lower() for c in cards}
# Pull quoted card names from FAQ heuristically
faq_quoted = re.findall(r'"([A-Z][A-Za-z \',-]{2,40})"', faq_text)
for fname in dict.fromkeys(faq_quoted):   # preserve order, deduplicate
    if fname.lower() not in our_names_lower:
        faq_card_names_not_found.append(fname)

# ---------------------------------------------------------------------------
# Write uprising_cards.json
# ---------------------------------------------------------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(cards, f, indent=2, ensure_ascii=False)
print(f"Saved JSON  â†’ {OUTPUT_JSON}")

# ---------------------------------------------------------------------------
# Write card_definitions.py
# ---------------------------------------------------------------------------
cards_by_key = {}
for card in cards:
    key = to_snake_case(card.get("name", card.get("_source_file", "unknown")))
    cards_by_key[key] = {k: v for k, v in card.items() if k != "_source_file"}

py_lines = [
    '"""',
    "Auto-generated card definitions for Dune Imperium: Uprising.",
    "Regenerate with:  python extract_cards.py",
    '"""',
    "",
    "from typing import Any, Dict",
    "",
    "UPRISING_CARDS: Dict[str, Dict[str, Any]] = " + json.dumps(cards_by_key, indent=4, ensure_ascii=False),
    "",
]
OUTPUT_PY.write_text("\n".join(py_lines), encoding="utf-8")
print(f"Saved Python â†’ {OUTPUT_PY}")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"  Total cards extracted:          {len(cards)}")
print(f"  Cards flagged for manual review: {len(failed)}")
print(f"  Cards annotated with FAQ notes:  {faq_annotated}")

if failed:
    print("\nFailed cards:")
    for entry in failed:
        print(f"  {entry['file']}: {entry['error']}")

if faq_card_names_not_found:
    print(f"\nCard names in FAQ not found in image folder ({len(faq_card_names_not_found)}):")
    for name in faq_card_names_not_found[:30]:
        print(f"  \"{name}\"")
    if len(faq_card_names_not_found) > 30:
        print(f"  ... and {len(faq_card_names_not_found) - 30} more")

