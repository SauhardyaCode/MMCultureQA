"""
Command-Line Interface (CLI) for SemEval 2027 MMCultureQA.
Provides subcommands for dataset management, inference, evaluation, and CodaBench submission.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

from .asr import SpeechTranscriber
from .config import Config, HF_DATASET_ID, PROJECT_ROOT, SUPPORTED_LANGUAGES
from .dataset import MMCultureQADataset
from .evaluation import CodaBenchPackager, MMCultureQAEvaluator
from .models import CulturalVLMSolver, LocalCulturalBaseline, MMCultureQAPipeline

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mmcultureqa.cli")


def cmd_download(args: argparse.Namespace) -> None:
    """Download or refresh official dataset from Hugging Face."""
    cfg = Config()
    cfg.ensure_directories()
    print(f"[*] Querying Hugging Face repository '{HF_DATASET_ID}'...")

    api_url = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "MMCultureQA-Prototype"})
        meta = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    except Exception as e:
        print(f"[!] Error querying Hugging Face API: {e}")
        return

    siblings = [s["rfilename"] for s in meta.get("siblings", [])]
    target_files = [f for f in siblings if f.startswith(("images/", "audio/", "qa/", "sqa/"))]

    print(f"[*] Found {len(target_files)} official dataset files.")
    base_raw = f"https://huggingface.co/datasets/{HF_DATASET_ID}/raw/main/"
    base_resolve = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/"

    downloaded = 0
    for f in target_files:
        dest_path = cfg.data_dir / f
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 0 and not args.force:
            continue

        # Use resolve URL for binary LFS assets (images, audio) and raw for jsonl
        url = base_resolve + f if f.startswith(("images/", "audio/")) else base_raw + f
        try:
            data = urllib.request.urlopen(url).read()
            dest_path.write_bytes(data)
            downloaded += 1
            print(f"    [+] Downloaded: {f} ({len(data)} bytes)")
        except Exception as e:
            print(f"    [!] Failed {f}: {e}")

    print(f"[OK] Download complete. {downloaded} new files saved into '{cfg.data_dir}'.")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect dataset splits, schemas, and cultural distribution."""
    cfg = Config()
    split = args.split or "train"
    task = args.task or "qa"
    lang = args.lang or "en"

    try:
        ds = MMCultureQADataset.load_split(split=split, task=task, lang=lang, data_dir=cfg.data_dir)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return

    stats = ds.summary_statistics()
    integrity = ds.validate_integrity()

    print("=" * 60)
    print(f" MMCultureQA Split Inspector: {split.upper()} | Task: {task.upper()} | Lang: {lang.upper()}")
    print("=" * 60)
    print(f"Total Records: {stats['total']}")
    print(f"Media Integrity: {'[VALID]' if integrity['valid'] else '[ISSUES DETECTED]'}")
    if not integrity["valid"]:
        print(f"  - Missing Images: {integrity['missing_images_count']}")
        print(f"  - Missing Audio:  {integrity['missing_audio_count']}")

    print("\n[+] Top Cultural Countries:")
    for country, count in stats["countries"][:5]:
        print(f"    * {country:15s}: {count} records")

    print("\n[+] Top Cultural Categories:")
    for category, count in stats["categories"][:5]:
        print(f"    * {category:30s}: {count} records")

    if len(ds) > 0:
        sample = ds[0]
        print("\n[+] Sample Record:")
        print(f"    * ID:          {sample.id}")
        print(f"    * Image:       {sample.image}")
        if sample.question:
            print(f"    * Question:    {sample.question}")
        if sample.audio:
            print(f"    * Audio:       {sample.audio}")
        if sample.answer:
            print(f"    * Reference:   {sample.answer}")
        print(f"    * Country:     {sample.country}")
        print(f"    * Category:    {sample.category} -> {sample.subcategory}")
    print("=" * 60)


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate culturally grounded predictions for single instances or datasets."""
    cfg = Config()

    # Select solver
    if args.solver == "vlm":
        print("[*] Initializing Cultural VLM Solver (RAG)...")
        try:
            solver = CulturalVLMSolver(model_name=args.model)
        except ValueError as e:
            print(f"[!] {e}")
            print("[*] Falling back to Local Cultural Baseline...")
            solver = LocalCulturalBaseline()
    else:
        print("[*] Initializing Local Cultural Baseline...")
        solver = LocalCulturalBaseline()

    pipeline = MMCultureQAPipeline(solver=solver)

    # Single instance prediction
    if args.image and (args.question or args.audio):
        print(f"[*] Running inference for single instance (lang: {args.lang})...")
        if args.audio:
            res = pipeline.run_task1_sqa(args.image, args.audio, lang=args.lang)
            print("\n[OK] Task 1 (Spoken QA) Result:")
            print(f"    Audio:         {res['audio']}")
            print(f"    Transcribed:   {res['transcription']}")
            print(f"    Answer:        {res['prediction']}\n")
        else:
            res = pipeline.run_task2_qa(args.image, args.question, lang=args.lang)
            print("\n[OK] Task 2 (Text QA) Result:")
            print(f"    Question:      {res['question']}")
            print(f"    Answer:        {res['prediction']}\n")
        return

    # Batch split prediction
    split = args.split or "dev"
    task = args.task or "qa"
    lang = args.lang or "en"
    print(f"[*] Running batch predictions for split='{split}', task='{task}', lang='{lang}'...")

    try:
        ds = MMCultureQADataset.load_split(split=split, task=task, lang=lang, data_dir=cfg.data_dir)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return

    out_file = args.output or (cfg.submissions_dir / f"predictions_{task}_{lang}.json")
    results = pipeline.predict_dataset(ds, output_file=out_file, limit=args.limit)
    print(f"[OK] Successfully generated {len(results)} predictions -> '{out_file}'")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate predictions against ground-truth references using official metrics."""
    cfg = Config()
    pred_path = Path(args.preds)
    if not pred_path.exists():
        print(f"[!] Predictions file not found: {pred_path}")
        return

    ref_path = Path(args.refs) if args.refs else (cfg.qa_dir / f"dev_{args.lang}.jsonl")
    if not ref_path.exists():
        print(f"[!] References file not found: {ref_path}")
        return

    print(f"[*] Evaluating predictions from '{pred_path}' against references '{ref_path}'...")
    evaluator = MMCultureQAEvaluator(pred_path, ref_path, lang=args.lang)
    results = evaluator.evaluate()

    # Print summary
    print("\n" + evaluator.format_summary_table(results) + "\n")

    if args.output:
        evaluator.export_report(results, args.output)
        print(f"[OK] Exported detailed evaluation report to '{args.output}'")


def cmd_submit(args: argparse.Namespace) -> None:
    """Package predictions into CodaBench submission.zip archive."""
    pred_path = Path(args.preds)
    if not pred_path.exists():
        print(f"[!] Predictions file not found: {pred_path}")
        return

    out_zip = Path(args.output or "submission.zip")
    print(f"[*] Packaging '{pred_path}' into CodaBench archive '{out_zip}'...")

    try:
        meta = CodaBenchPackager.create_submission_zip(pred_path, out_zip)
        print(f"[OK] CodaBench submission archive successfully created!")
        print(f"    * Archive:     {meta['submission_zip']}")
        print(f"    * Records:     {meta['records_count']}")
        print(f"    * Size:        {meta['size_bytes']} bytes")
        print(f"    * Content:     {meta['contains_file']}")
        print("\nReady for upload to CodaBench!")
    except Exception as e:
        print(f"[!] Submission packaging failed: {e}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Launch interactive Streamlit web dashboard."""
    app_file = PROJECT_ROOT / "app.py"
    if not app_file.exists():
        print(f"[!] app.py not found at {app_file}")
        return

    port = args.port or 8501
    print(f"[*] Launching Streamlit web application on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file), "--server.port", str(port)])


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mmcultureqa",
        description="MMCultureQA 2027: SemEval Shared Task Multilingual Multimodal Cultural QA Prototype",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Download
    p_dl = subparsers.add_parser("download", help="Download official dataset from Hugging Face")
    p_dl.add_argument("--force", action="store_true", help="Overwrite existing files")

    # Inspect
    p_ins = subparsers.add_parser("inspect", help="Inspect dataset split, schema, and statistics")
    p_ins.add_argument("--split", default="train", help="Split to inspect (train or dev)")
    p_ins.add_argument("--task", default="qa", choices=["qa", "sqa"], help="Task ('qa' or 'sqa')")
    p_ins.add_argument("--lang", default="en", help="Language code (e.g., 'en', 'msa')")

    # Predict
    p_pred = subparsers.add_parser("predict", help="Generate predictions for instance or split")
    p_pred.add_argument("--image", help="Path to input image file")
    p_pred.add_argument("--question", help="Question text (for Task 2)")
    p_pred.add_argument("--audio", help="Path to spoken question WAV file (for Task 1)")
    p_pred.add_argument("--split", default="dev", help="Dataset split ('train' or 'dev')")
    p_pred.add_argument("--task", default="qa", choices=["qa", "sqa"], help="Task ('qa' or 'sqa')")
    p_pred.add_argument("--lang", default="en", help="Language track code")
    p_pred.add_argument("--solver", default="local", choices=["local", "vlm"], help="Solver engine")
    p_pred.add_argument("--model", default="gpt-4o-mini", help="VLM model identifier")
    p_pred.add_argument("--limit", type=int, help="Limit number of records to predict")
    p_pred.add_argument("--output", help="Output JSON path for predictions")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute BERTScore and auxiliary metrics")
    p_eval.add_argument("--preds", required=True, help="Path to predictions JSON file")
    p_eval.add_argument("--refs", help="Path to ground-truth references JSONL file")
    p_eval.add_argument("--lang", default="en", help="Evaluation language code")
    p_eval.add_argument("--output", help="Optional JSON output report path")

    # Submit
    p_sub = subparsers.add_parser("submit", help="Package predictions into CodaBench submission zip")
    p_sub.add_argument("--preds", required=True, help="Path to predictions JSON file")
    p_sub.add_argument("--output", default="submission.zip", help="Path to output submission.zip")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Launch interactive Streamlit web dashboard")
    p_demo.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "download": cmd_download,
        "inspect": cmd_inspect,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "submit": cmd_submit,
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
