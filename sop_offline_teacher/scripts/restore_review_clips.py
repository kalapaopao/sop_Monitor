"""Rebuild optional review media from committed manifests and local source videos.

No inference or network calls. Existing media are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web" / "public"
MANIFESTS = (
    "sop-discovery/clips.json", "sop-refinement/clips.json",
    "sop-full/segments.json", "three-workers-examples/clips.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--manifest", action="append", help="Path relative to web/public")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = {}
    for item in args.video:
        key, sep, value = item.partition("=")
        if not key or not sep or not value:
            parser.error("--video requires ID=PATH")
        sources[key] = Path(value).resolve()

    clips = []
    for name in args.manifest or MANIFESTS:
        manifest = (PUBLIC / name).resolve()
        if not manifest.is_relative_to(PUBLIC.resolve()):
            parser.error("manifest must stay inside web/public")
        doc = json.loads(manifest.read_text(encoding="utf-8-sig"))
        clips.extend(doc.get("clips", doc.get("segments", [])))
    clips = [c for c in clips if c.get("url")]
    if args.limit:
        clips = clips[:args.limit]
    jobs = []
    for clip in clips:
        output = (PUBLIC / clip["url"].lstrip("/")).resolve()
        if not output.is_relative_to(PUBLIC.resolve()) or output.suffix != ".mp4":
            parser.error("invalid media URL in manifest")
        if output.exists() and output.with_suffix(".jpg").exists():
            continue
        source = sources.get(clip["video"])
        if not output.exists() and (source is None or not source.is_file()):
            parser.error(f"missing local source: --video {clip['video']}=PATH")
        start, end = int(clip["start_frame"]), int(clip["end_frame"])
        if start < 0 or end <= start:
            parser.error("invalid frame interval in manifest")
        jobs.append((source, output, start, end))
    print(f"{len(clips)} manifest clips; {len(jobs)} need media; dry_run={args.dry_run}")
    if args.dry_run or not jobs:
        return 0
    executable = shutil.which(args.ffmpeg)
    if executable is None:
        parser.error("ffmpeg not found; supply --ffmpeg PATH")
    for source, output, start, end in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            # Decode in original frame order; trim preserves the manifest's frame bounds.
            subprocess.run([
                executable, "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
                "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                "-frames:v", str(end - start), "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                str(output),
            ], check=True)
        poster = output.with_suffix(".jpg")
        if not poster.exists():
            subprocess.run([
                executable, "-hide_banner", "-loglevel", "error", "-n", "-i", str(output),
                "-frames:v", "1", "-update", "1", str(poster),
            ], check=True)
        print(output.relative_to(PUBLIC))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
