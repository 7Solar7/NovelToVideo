"""
Standalone audio generation tool.
Generates TTS narration + SRT subtitles from text without GPU/image pipeline.
Usage:
    python generate_audio.py chapter1.txt
    python generate_audio.py chapter1.txt --voice en-US-JennyNeural --rate -10% --output custom_name
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, TTS_VOICE, TTS_RATE, TTS_VOLUME
from tts_generator import run_tts


def read_input_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(description="Generate TTS narration audio + SRT subtitles from text")
    parser.add_argument("input_text", help="Path to input .txt file")
    parser.add_argument("output_name", nargs="?", default=None,
                        help="Output filename base (default: input filename without extension)")
    parser.add_argument("--voice", default=TTS_VOICE,
                        help=f"edge-tts voice (default: {TTS_VOICE})")
    parser.add_argument("--rate", default=TTS_RATE,
                        help=f"Speaking rate (default: {TTS_RATE})")
    parser.add_argument("--volume", default=TTS_VOLUME,
                        help=f"Volume adjustment (default: {TTS_VOLUME})")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available edge-tts voices and exit")
    args = parser.parse_args()

    if args.list_voices:
        import asyncio
        import edge_tts
        async def list_voices():
            voices = await edge_tts.list_voices()
            print("\nAvailable voices:")
            for v in voices:
                print(f"  {v['ShortName']:30s} | {v['Gender']:6s} | {v['Locale']}")
        asyncio.run(list_voices())
        return

    output_base = args.output_name or os.path.splitext(os.path.basename(args.input_text))[0]

    print("=" * 50)
    print("TTS NARRATION + SUBTITLE GENERATOR")
    print("=" * 50)

    print(f"\n[1/2] Reading text: {args.input_text}")
    text = read_input_text(args.input_text)
    print(f"Loaded {len(text)} characters")

    print(f"\n[2/2] Generating TTS (voice: {args.voice}, rate: {args.rate})...")
    try:
        result = run_tts(text, output_base)
        print(f"\nOutput files:")
        print(f"  Audio:     {result['audio_path']}")
        print(f"  Subtitles: {result['srt_path']}")
        print(f"\nDone! {len(text)} characters narrated in {len(result['srt_content'].splitlines()) // 4} subtitle entries.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
