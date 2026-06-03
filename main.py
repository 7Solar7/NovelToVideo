import json
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, BGM_PATH, SEGMENT_METHOD, STYLE_PREFIX, CHARACTER_SHEET_PATH
from tts_generator import run_tts
from scene_analyzer import analyze_text, extract_character_sheet, split_sentences, merge_sentences
from image_generator import load_pipeline, generate_images
from video_assembler import create_video_with_ffmpeg_dub


def read_input_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main(input_text_path: str, output_name: str = "final_output", character_sheet_path: str = None):
    if not os.path.exists(input_text_path):
        print(f"ERROR: Input file not found: {input_text_path}")
        sys.exit(1)

    print("=" * 50)
    print("NOVEL-TO-VIDEO PIPELINE")
    print("=" * 50)

    # ----- Stage 1: Read input -----
    print("\n[1/5] Reading input text...")
    full_text = read_input_text(input_text_path)
    print(f"Loaded {len(full_text)} characters")

    # ----- Stage 1.5: Character sheet -----
    character_sheet = None
    if character_sheet_path:
        print(f"\n[1.5/5] Loading character sheet from: {character_sheet_path}")
        with open(character_sheet_path, "r", encoding="utf-8") as f:
            character_sheet = json.load(f)
        print(f"Loaded {len(character_sheet)} character(s)")
    else:
        print(f"\n[1.5/5] Extracting character descriptions from text (Ollama)...")
        try:
            character_sheet = extract_character_sheet(full_text)
            if character_sheet:
                with open(CHARACTER_SHEET_PATH, "w", encoding="utf-8") as f:
                    json.dump(character_sheet, f, indent=2)
                print(f"Extracted {len(character_sheet)} character(s), saved to {CHARACTER_SHEET_PATH}")
            else:
                print("No named characters found in text")
        except Exception as e:
            print(f"Character extraction failed (non-fatal): {e}")
            character_sheet = None

    # ----- Stage 2: Scene Analysis (Ollama) -----
    print("\n[2/5] Analyzing text and generating image prompts (Ollama)...")
    try:
        sentences_per_image = 2 if SEGMENT_METHOD == "sentence" else 3
        segments = analyze_text(full_text, sentences_per_image=sentences_per_image, character_sheet=character_sheet)
    except Exception as e:
        print(f"ERROR in scene analysis: {e}")
        print("Falling back to sentence-based segmentation without prompts...")
        sentences = split_sentences(full_text)
        groups = merge_sentences(sentences, sentences_per_image)
        segments = [
            {"index": i, "text": g, "prompt": STYLE_PREFIX + g}
            for i, g in enumerate(groups)
        ]
    print(f"Generated {len(segments)} segments with image prompts")

    # ----- Stage 3: TTS + Subtitles (edge-tts) -----
    print("\n[3/5] Generating TTS narration and subtitles (edge-tts)...")
    try:
        tts_result = run_tts(full_text, output_name)
    except Exception as e:
        print(f"ERROR in TTS generation: {e}")
        sys.exit(1)

    # ----- Stage 4: Image Generation (diffusers) -----
    print("\n[4/5] Generating images (Stable Diffusion)...")
    pipe = None
    try:
        pipe = load_pipeline()
        image_paths = generate_images(segments, pipe=pipe)
    except Exception as e:
        print(f"ERROR in image generation: {e}")
        sys.exit(1)
    finally:
        if pipe is not None:
            del pipe
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Generated/cached {len(image_paths)} images")

    # ----- Stage 5: Video Assembly (MoviePy + FFmpeg) -----
    print("\n[5/5] Assembling final video...")
    final_output = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
    bgm_path = BGM_PATH if os.path.exists(BGM_PATH) else None

    try:
        result = create_video_with_ffmpeg_dub(
            image_paths=image_paths,
            narration_audio_path=tts_result["audio_path"],
            srt_path=tts_result["srt_path"],
            output_path=final_output,
            bgm_path=bgm_path,
        )
    except Exception as e:
        print(f"ERROR in video assembly: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print(f"Output: {final_output}")
    print("=" * 50)

    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel-to-video pipeline")
    parser.add_argument("input_text_path", help="Path to input text file")
    parser.add_argument("output_name", nargs="?", default="final_output", help="Output video name (default: final_output)")
    parser.add_argument("--characters", help="Path to character sheet JSON from a previous run")
    args = parser.parse_args()

    main(args.input_text_path, args.output_name, character_sheet_path=args.characters)
