"""Render and review only PRML 3.4 or 5.1, using their existing validators."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = {
    "3.4": ("3.*/3.4.*", "prml_3_4_bayesian_model_comparison", "PRML34BayesianModelComparison", "prml34"),
    "5.1": ("5.*/5.1.*", "prml_5_1_feed_forward_network_functions", "PRML51FeedForwardNetworkFunctions", "prml51"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=SECTIONS, required=True)
    parser.add_argument("--quality", choices=["low", "high"], default="low")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    pattern, module, scene, prefix = SECTIONS[args.section]
    directory, = (ROOT / "feature").glob(pattern)
    destination = ROOT / "review-output" / args.section
    destination.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    resolution = "480p15" if args.quality == "low" else "1080p60"
    video = directory / "media" / "videos" / module / resolution / (scene + ".mp4")
    if not args.review_only:
        subprocess.run([sys.executable, "verify_numerics.py", "--captions"], check=True)
        subprocess.run([sys.executable, "-m", "manim", "--progress_bar", "none",
                        "--disable_caching", "--flush_cache", "-ql" if args.quality == "low" else "-qh",
                        module + ".py", scene], check=True)
    spec = importlib.util.spec_from_file_location("review_video", directory / "verify_video.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.VIDEO = video
    sys.argv = ["verify_video.py", "--out" if args.section == "3.4" else "--output", str(destination)]
    validator.main()
    shutil.copy2(video, destination / video.name)
    timeline = directory / "media" / (prefix + "_timeline.json")
    shutil.copy2(timeline, destination / timeline.name)
    if args.section == "5.1":
        shutil.copy2(directory / "validation_results.json", destination / "review.json")
    # Explicit samples around the reported defects, in addition to the existing review.
    times = [139.7, 140.0, 140.3, 144.0, 146.1, 146.5, 150.0, 151.0] if args.section == "3.4" else [4, 6, 8, 10, 12, 13.8, 14.2]
    for seconds in times:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(seconds), "-i", str(video),
                        "-frames:v", "1", str(destination / f"defect-{seconds:.1f}.png")], check=True)
    print(json.dumps({"section": args.section, "video": str(video), "review": str(destination)}))


if __name__ == "__main__":
    main()
