import asyncio
import os
import edge_tts
from config import TTS_VOICE, TTS_RATE, TTS_VOLUME, OUTPUT_DIR


def generate_srt_from_submaker(submaker) -> str:
    return submaker.get_srt()


async def generate_tts(text: str, filename_base: str = "narration") -> dict:
    audio_path = os.path.join(OUTPUT_DIR, f"{filename_base}.mp3")
    srt_path = os.path.join(OUTPUT_DIR, f"{filename_base}.srt")

    communicate = edge_tts.Communicate(
        text,
        TTS_VOICE,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
    )
    submaker = edge_tts.SubMaker()

    with open(audio_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)

    srt_content = generate_srt_from_submaker(submaker)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"TTS audio saved: {audio_path}")
    print(f"SRT subtitles saved: {srt_path}")

    return {
        "audio_path": audio_path,
        "srt_path": srt_path,
        "srt_content": srt_content,
    }


def run_tts(text: str, filename_base: str = "narration") -> dict:
    return asyncio.run(generate_tts(text, filename_base))


if __name__ == "__main__":
    test_text = "The warrior stepped into the light. The ancient forest whispered secrets. A dragon soared across the crimson sky."
    result = run_tts(test_text, "test_narration")
    print(result["srt_content"])
