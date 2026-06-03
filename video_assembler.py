import os
import re
import shutil
import subprocess
import tempfile
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from config import (
    OUTPUT_DIR, OUTPUT_FPS, BGM_PATH, BGM_VOLUME,
    DUCK_THRESHOLD, DUCK_RATIO, DUCK_ATTACK, DUCK_RELEASE,
    KEN_BURNS_ZOOM_SPEED, SD_WIDTH, SD_HEIGHT
)


def parse_srt(srt_content: str):
    pattern = re.compile(
        r"(\d+)\n(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.+?)(?=\n\n|\Z)",
        re.DOTALL
    )
    segments = []
    for match in pattern.finditer(srt_content):
        start = _srt_time_to_seconds(match.group(2))
        end = _srt_time_to_seconds(match.group(3))
        text = match.group(4).replace("\n", " ").strip()
        segments.append({"start": start, "end": end, "text": text, "duration": end - start})
    return segments


def _srt_time_to_seconds(t):
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def create_video_with_ffmpeg_dub(
    image_paths,
    narration_audio_path,
    srt_path,
    output_path,
    bgm_path=None,
):
    temp_video = os.path.join(OUTPUT_DIR, "_temp_no_audio.mp4")
    temp_mixed_audio = os.path.join(OUTPUT_DIR, "_temp_mixed_audio.wav")

    clips = []
    image_paths_sorted = sorted(image_paths)

    if not os.path.exists(narration_audio_path):
        print(f"ERROR: Narration audio not found: {narration_audio_path}")
        return None

    narration = AudioFileClip(narration_audio_path)

    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    srt_segments = parse_srt(srt_content)
    num_images = len(image_paths_sorted)

    entries_per_image = len(srt_segments) / num_images
    print(f"Distributing {len(srt_segments)} SRT entries across {num_images} images ({entries_per_image:.2f} per image)")

    last_end = 0
    image_durations = []
    for i in range(num_images):
        start_idx = round(i * entries_per_image)
        end_idx = round((i + 1) * entries_per_image)
        if start_idx < last_end:
            start_idx = last_end
        if end_idx > len(srt_segments):
            end_idx = len(srt_segments)
        duration = sum(seg["duration"] for seg in srt_segments[start_idx:end_idx])
        image_durations.append(duration)
        last_end = end_idx

    for i in range(num_images):
        img_path = image_paths_sorted[i]
        dur = image_durations[i]

        clip = ImageClip(img_path, duration=dur)
        clip = clip.resize(lambda t: 1 + KEN_BURNS_ZOOM_SPEED * t)
        clip = clip.set_position(("center", "center"))
        clips.append(clip)

    print("Concatenating image clips...")
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_fps(OUTPUT_FPS)

    print("Writing temp video (no audio)...")
    final_clip.write_videofile(
        temp_video,
        fps=OUTPUT_FPS,
        codec="libx264",
        audio=False,
        logger=None,
    )

    final_clip.close()
    for c in clips:
        c.close()

    print("Mixing audio with FFmpeg (sidechaincompress ducking)...")
    if bgm_path and os.path.exists(bgm_path):
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", narration_audio_path,
            "-i", bgm_path,
            "-filter_complex",
            f"[1:a]volume={BGM_VOLUME}[bgm];"
            f"[bgm][0:a]sidechaincompress="
            f"threshold={DUCK_THRESHOLD}:"
            f"ratio={DUCK_RATIO}:"
            f"attack={DUCK_ATTACK}:"
            f"release={DUCK_RELEASE}[ducked]",
            "-map", "[ducked]",
            temp_mixed_audio,
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", narration_audio_path,
            "-c", "copy",
            temp_mixed_audio,
        ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    print("Burning subtitles and combining audio with FFmpeg...")
    tmp_srt_name = "novel_to_video_subtitles.srt"
    tmp_srt = os.path.join(tempfile.gettempdir(), tmp_srt_name)
    shutil.copy2(srt_path, tmp_srt)

    # Use relative filename via cwd to avoid drive-letter colon in subtitles filter
    final_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", temp_mixed_audio,
        "-vf", f"subtitles={tmp_srt_name}:force_style='Fontsize=18,PrimaryColour=&H00FFFFFF,BackColour=&HA0000000,BorderStyle=4'",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    subprocess.run(final_cmd, check=True, capture_output=True, cwd=tempfile.gettempdir())

    print(f"Final video saved: {output_path}")

    for tmp in [temp_video, temp_mixed_audio, tmp_srt]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return output_path
