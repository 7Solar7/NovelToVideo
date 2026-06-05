import re
from config import OLLAMA_MODEL, OLLAMA_HOST, STYLE_PREFIX, FULL_STYLE_DESCRIPTION, TRIGGER_WORD


def _get_ollama_client():
    try:
        import ollama
        return ollama
    except ImportError:
        print("ERROR: 'ollama' Python package not installed. Run: pip install ollama")
        raise


def split_sentences(text: str):
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def merge_sentences(sentences, max_group_size=3):
    groups = []
    for i in range(0, len(sentences), max_group_size):
        group = sentences[i:i + max_group_size]
        groups.append(" ".join(group))
    return groups


def generate_prompt(segment_text: str, character_sheet: dict = None) -> str:
    prompt = (
        f"You are a visual scene director. For the following narration segment, "
        f"create a detailed visual prompt for an AI image generator.\n\n"
        f"The visual style must match:\n{FULL_STYLE_DESCRIPTION}\n\n"
        f"IMPORTANT: Include the style trigger word '{TRIGGER_WORD}' naturally in your prompt.\n\n"
        f"Describe only the visual scene — characters, setting, lighting, colors, "
        f"composition. Keep it concise (under 35 words).\n\n"
        f"Narration: '{segment_text}'\n\nVisual prompt:"
    )

    if character_sheet:
        char_lines = "\n".join(
            f"  - {name}: {desc}" for name, desc in character_sheet.items()
        )
        prompt += (
            f"\n\nReference character appearances (if any of these characters "
            f"appear in the scene, naturally describe their relevant visual traits):\n"
            f"{char_lines}"
        )

    ollama = _get_ollama_client()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_prompt = response["message"]["content"].strip()
    cleaned = clean_prompt(raw_prompt)
    if TRIGGER_WORD not in cleaned.lower():
        cleaned = f"{TRIGGER_WORD}, {cleaned}"
    return STYLE_PREFIX + cleaned


def clean_prompt(raw: str) -> str:
    lines = raw.replace("\n", " ").split(".")
    cleaned = []
    for part in lines:
        part = part.strip()
        lower = part.lower()
        if any(lower.startswith(p) for p in [
            "here is", "here's", "visual prompt", "prompt:",
            "image prompt", "**", "note:", "description:",
        ]):
            continue
        if part:
            cleaned.append(part)
    result = ". ".join(cleaned)
    tokens = result.split()
    if len(tokens) > 35:
        result = " ".join(tokens[:35]) + "."
    return result


def parse_json_from_ollama(raw: str) -> dict:
    import json
    code_block = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if code_block:
        raw = code_block.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("WARNING: Could not parse character sheet JSON from Ollama.")
        print(f"Raw: {raw[:300]}...")
        return {}


def extract_character_sheet(text: str) -> dict:
    prompt = (
        "From the following novel text, identify every distinct named character. "
        "For each character, describe their consistent physical appearance: "
        "hair color/style, eye color, build, typical clothing, "
        "distinguishing features.\n\n"
        "Return ONLY valid JSON. Example:\n"
        '{"Ethan": "tall, broad-shouldered, short brown hair, green eyes, '
        'leather jacket",\n'
        ' "Lena": "slim, waist-length black hair, pale blue eyes, dark coat"}\n\n'
        f"Text:\n{text[:6000]}"
    )

    ollama = _get_ollama_client()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_json_from_ollama(response["message"]["content"].strip())


def analyze_text(text: str, sentences_per_image: int = 2, character_sheet: dict = None) -> list:
    sentences = split_sentences(text)
    print(f"Detected {len(sentences)} sentences")

    merged = merge_sentences(sentences, sentences_per_image)
    print(f"Grouped into {len(merged)} segments for image generation")

    if character_sheet:
        print(f"Loaded {len(character_sheet)} characters for reference injection")

    segments = []
    for i, group_text in enumerate(merged):
        print(f"Generating prompt for segment {i + 1}/{len(merged)}...")
        image_prompt = generate_prompt(group_text, character_sheet=character_sheet)
        segments.append({
            "index": i,
            "text": group_text,
            "prompt": image_prompt,
        })

    return segments


if __name__ == "__main__":
    test = (
        "The warrior stepped into the ancient forest. "
        "Moonlight filtered through the dense canopy above. "
        "A dragon's roar echoed in the distance. "
        "The ground trembled beneath his feet. "
        "He gripped his sword and moved forward."
    )
    result = analyze_text(test, sentences_per_image=2)
    for seg in result:
        print(f"\nSegment {seg['index']}:")
        print(f"  Text: {seg['text'][:60]}...")
        print(f"  Prompt: {seg['prompt'][:80]}...")
