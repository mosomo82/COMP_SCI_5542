"""
evaluate_quality.py
-------------------
Interactive review tool for manually rating generated images.

Features
--------
- Loads images from results/evaluation_report.csv when available
- Falls back to scanning outputs/ recursively for PNG files
- Displays one image at a time in a Tkinter window
- Captures 1-5 quality scores plus optional reviewer notes
- Supports resume from existing results/quality_scores.csv
- Writes results/quality_scores.csv and results/quality_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk
from rich.console import Console


console = Console()

ROOT_DIR = Path(__file__).parent
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_EVAL_REPORT = DEFAULT_RESULTS_DIR / "evaluation_report.csv"
DEFAULT_QUALITY_FILE = DEFAULT_RESULTS_DIR / "quality_scores.csv"
DEFAULT_QUALITY_SUMMARY = DEFAULT_RESULTS_DIR / "quality_summary.csv"

MAX_IMAGE_SIZE = (720, 720)


@dataclass
class ReviewItem:
    product_id: str
    product_title: str
    prompt_type: str
    image_path: str
    prompt: str

    @property
    def image_name(self) -> str:
        return Path(self.image_path).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive qualitative review for generated product images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS_DIR), help="Directory containing generated images")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for evaluation and quality CSV files")
    parser.add_argument("--report", default=str(DEFAULT_EVAL_REPORT), help="Optional evaluation_report.csv path used to define review order")
    parser.add_argument("--start-at", type=int, default=0, help="Start review at a specific 0-based index")
    return parser.parse_args()


def _workspace_relative(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_image_path(path_str: str, outputs_dir: Path) -> Path:
    raw_path = Path(path_str)
    if raw_path.exists():
        return raw_path

    if not raw_path.is_absolute():
        candidate = ROOT_DIR / raw_path
        if candidate.exists():
            return candidate

    if raw_path.name:
        matches = list(outputs_dir.rglob(raw_path.name))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Image not found: {path_str}")


def load_review_items(report_path: Path, outputs_dir: Path) -> list[ReviewItem]:
    items: list[ReviewItem] = []

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    resolved = _resolve_image_path(row["image_path"], outputs_dir)
                except FileNotFoundError:
                    continue
                items.append(
                    ReviewItem(
                        product_id=row.get("product_id", ""),
                        product_title=row.get("product_title", ""),
                        prompt_type=row.get("prompt_type", ""),
                        image_path=str(resolved),
                        prompt=row.get("prompt", ""),
                    )
                )
        if items:
            return items

    for png_path in sorted(outputs_dir.rglob("*.png")):
        image_name = png_path.stem
        parts = image_name.split("_")
        product_id = parts[0] if parts else "unknown"
        prompt_type = parts[1] if len(parts) > 1 else "unknown"
        items.append(
            ReviewItem(
                product_id=product_id,
                product_title=product_id,
                prompt_type=prompt_type,
                image_path=str(png_path),
                prompt="",
            )
        )
    return items


def load_existing_scores(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            image_name = row.get("image_name") or Path(row.get("image_path", "")).name
            if image_name:
                rows[image_name] = row
        return rows


def save_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product_id",
        "product_title",
        "prompt_type",
        "image_name",
        "image_path",
        "quality_score",
        "quality_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(scores_path: Path, summary_path: Path) -> None:
    if not scores_path.exists():
        return

    df = pd.read_csv(scores_path)
    if df.empty:
        return

    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    grouped = (
        df.groupby(["product_id", "product_title", "prompt_type"], dropna=False)
        .agg(
            mean_quality_score=("quality_score", "mean"),
            n_quality_scored=("quality_score", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    grouped["mean_quality_score"] = grouped["mean_quality_score"].round(2)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(summary_path, index=False)


class QualityReviewApp:
    def __init__(self, root: tk.Tk, items: list[ReviewItem], scores_path: Path, summary_path: Path, start_at: int = 0):
        self.root = root
        self.items = items
        self.scores_path = scores_path
        self.summary_path = summary_path
        self.scores = load_existing_scores(scores_path)
        self.index = max(0, min(start_at, len(items) - 1)) if items else 0
        self.photo: ImageTk.PhotoImage | None = None

        self.root.title("Quality Review")
        self.root.geometry("980x980")

        self.score_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.path_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()
        self._render_current()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(frame, text="Qualitative Quality Scoring", font=("Segoe UI", 18, "bold"))
        header.pack(anchor="w", pady=(0, 8))

        ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 6))
        ttk.Label(frame, textvariable=self.meta_var, font=("Segoe UI", 11, "bold"), wraplength=920).pack(anchor="w", pady=(0, 4))
        ttk.Label(frame, textvariable=self.path_var, font=("Consolas", 9), wraplength=920).pack(anchor="w", pady=(0, 10))

        self.image_label = ttk.Label(frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Overall quality score (1-5):", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        button_frame = ttk.Frame(controls)
        button_frame.grid(row=0, column=1, sticky="w", padx=(10, 0))
        for value in range(1, 6):
            ttk.Radiobutton(button_frame, text=str(value), value=str(value), variable=self.score_var).pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="Notes:", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="nw", pady=(12, 0))
        self.notes_entry = tk.Text(controls, width=80, height=5, wrap="word")
        self.notes_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(12, 0))
        controls.columnconfigure(1, weight=1)

        action_row = ttk.Frame(frame)
        action_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(action_row, text="Previous", command=self.prev_item).pack(side=tk.LEFT)
        ttk.Button(action_row, text="Save", command=self.save_current).pack(side=tk.LEFT, padx=8)
        ttk.Button(action_row, text="Save + Next", command=self.save_and_next).pack(side=tk.LEFT)
        ttk.Button(action_row, text="Skip", command=self.next_item).pack(side=tk.LEFT, padx=8)
        ttk.Button(action_row, text="Export Summary", command=self.export_summary).pack(side=tk.LEFT)

        help_text = "Shortcuts: 1-5 score | Ctrl+S save | Right/Enter next | Left previous"
        ttk.Label(frame, text=help_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 0))

    def _bind_keys(self) -> None:
        for value in range(1, 6):
            self.root.bind(str(value), lambda event, score=value: self.score_var.set(str(score)))
        self.root.bind("<Control-s>", lambda event: self.save_current())
        self.root.bind("<Return>", lambda event: self.save_and_next())
        self.root.bind("<Right>", lambda event: self.next_item())
        self.root.bind("<Left>", lambda event: self.prev_item())

    def _current_item(self) -> ReviewItem:
        return self.items[self.index]

    def _render_current(self) -> None:
        if not self.items:
            self.status_var.set("No images found.")
            return

        item = self._current_item()
        self.status_var.set(f"Image {self.index + 1} / {len(self.items)}")
        self.meta_var.set(f"{item.product_id} | {item.prompt_type} | {item.product_title}")
        self.path_var.set(_workspace_relative(item.image_path))

        image = Image.open(item.image_path).convert("RGB")
        image.thumbnail(MAX_IMAGE_SIZE)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        existing = self.scores.get(item.image_name, {})
        self.score_var.set(existing.get("quality_score", ""))
        self.notes_entry.delete("1.0", tk.END)
        self.notes_entry.insert("1.0", existing.get("quality_notes", ""))

    def _collect_row(self) -> dict:
        item = self._current_item()
        notes = self.notes_entry.get("1.0", tk.END).strip()
        return {
            "product_id": item.product_id,
            "product_title": item.product_title,
            "prompt_type": item.prompt_type,
            "image_name": item.image_name,
            "image_path": _workspace_relative(item.image_path),
            "quality_score": self.score_var.get().strip(),
            "quality_notes": notes,
        }

    def _persist(self) -> None:
        rows = sorted(self.scores.values(), key=lambda row: (row["product_id"], row["prompt_type"], row["image_name"]))
        save_scores(self.scores_path, rows)
        save_summary(self.scores_path, self.summary_path)

    def save_current(self) -> None:
        row = self._collect_row()
        if not row["quality_score"]:
            messagebox.showwarning("Missing score", "Choose a quality score from 1 to 5 before saving.")
            return
        self.scores[row["image_name"]] = row
        self._persist()
        self.status_var.set(f"Saved image {self.index + 1} / {len(self.items)}")

    def save_and_next(self) -> None:
        previous_index = self.index
        self.save_current()
        if self.index == previous_index and self.index < len(self.items) - 1:
            self.index += 1
            self._render_current()

    def next_item(self) -> None:
        if self.index < len(self.items) - 1:
            self.index += 1
            self._render_current()

    def prev_item(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def export_summary(self) -> None:
        self._persist()
        messagebox.showinfo("Summary exported", f"Saved:\n{self.scores_path}\n{self.summary_path}")


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    report_path = Path(args.report)
    scores_path = results_dir / DEFAULT_QUALITY_FILE.name
    summary_path = results_dir / DEFAULT_QUALITY_SUMMARY.name

    items = load_review_items(report_path, outputs_dir)
    if not items:
        console.print("[red]No images found for review.[/red]")
        raise SystemExit(1)

    console.print(f"[cyan]Loaded {len(items)} images for qualitative review.[/cyan]")
    console.print(f"[cyan]Scores will be saved to:[/cyan] {scores_path}")

    root = tk.Tk()
    QualityReviewApp(root, items, scores_path, summary_path, start_at=args.start_at)
    root.mainloop()


if __name__ == "__main__":
    main()