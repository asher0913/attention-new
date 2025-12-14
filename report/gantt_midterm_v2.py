#!/usr/bin/env python3
"""
Generate Gantt chart in the same visual style as the original proposal figure.
Planned vs actual/progress dates are encoded below; adjust as needed, then run:
    python gantt_midterm_v2.py
Output: artifacts/gantt_midterm_v2.png
"""

import os
from datetime import datetime, date
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Bars: (label, start, end, is_group, color_tag)
# color_tag: "plan" -> planned window; "actual" -> actual/progress
BARS = [
    ("1. Preliminary & proposal", "2024-10-01", "2024-11-30", True, "plan"),
    ("1.1 Proposal & scoping", "2024-10-01", "2024-10-31", False, "actual"),
    ("1.2 Ethics/risks prep", "2024-10-15", "2024-11-30", False, "actual"),

    ("2. Literature review", "2024-10-15", "2024-12-31", True, "plan"),
    ("2.1 Collaborative inference survey", "2024-10-15", "2024-11-15", False, "actual"),
    ("2.2 Attention/entropy surrogate study", "2024-11-01", "2024-12-15", False, "actual"),
    ("2.3 Baseline CEM reproduction", "2024-12-01", "2024-12-31", False, "actual"),

    ("3. Design & implementation", "2025-01-01", "2025-02-28", True, "plan"),
    ("3.1 Gated-attention surrogate", "2025-01-01", "2025-01-31", False, "actual"),
    ("3.2 Slot+gated cross-attn surrogate", "2025-02-01", "2025-02-28", False, "actual"),

    ("4. Experiments & analysis", "2025-02-01", "2025-03-31", True, "plan"),
    ("4.1 CIFAR-10 baseline/gated/slot runs", "2025-02-01", "2025-03-20", False, "actual"),
    ("4.2 Figures/tables/attack curves", "2025-02-20", "2025-03-31", False, "actual"),

    ("5. Fusion & ablations", "2025-03-15", "2025-04-15", True, "plan"),
    ("5.1 Slot–gate fusion prototypes", "2025-03-20", "2025-04-10", False, "actual"),
    ("5.2 Hyperparameter sweeps", "2025-03-25", "2025-04-15", False, "actual"),

    ("6. Extended evaluation", "2025-04-15", "2025-04-30", True, "plan"),
    ("6.1 Alt cuts / datasets", "2025-04-20", "2025-04-30", False, "plan"),

    ("7. Write-up & polish", "2025-05-01", "2025-05-31", True, "plan"),
    ("7.1 Final report & figures", "2025-05-01", "2025-05-31", False, "plan"),
]

MILESTONES = [
    ("Midterm report", "2025-03-31"),
    ("Final report", "2025-05-31"),
]

COLORS = {
    "plan": "#c7c7c7",
    "completed": "#55a868",
    "progress": "#dd8452",
}


def main(outfile="artifacts/gantt_midterm_v2.png"):
    os.environ.setdefault("MPLCONFIGDIR", "./tmp_mpl")
    fig, ax = plt.subplots(figsize=(12, 7))
    today = datetime.today().date()

    y_base, h_group, h_task, gap = 0, 0.7, 0.5, 0.25
    y = 0
    seen_labels = set()
    for label, start_str, end_str, is_group, tag in reversed(BARS):
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        width = (end_date - start_date).days + 1
        height = h_group if is_group else h_task
        if tag == "plan":
            color = COLORS["plan"]
            legend_lbl = "Planned"
        else:
            # actual/progress: color by completion
            color = COLORS["completed"] if end_date.date() <= today else COLORS["progress"]
            legend_lbl = "Completed" if color == COLORS["completed"] else "In progress"
        ax.barh(y, width, left=start_date, color=color, height=height, edgecolor="none", alpha=0.95)
        ax.text(start_date, y, label, va="center", ha="left",
                fontsize=9, fontweight="bold" if is_group else "normal", color="#1f1f1f")
        if legend_lbl not in seen_labels:
            seen_labels.add(legend_lbl)
        y += height + gap

    # Legend handles
    handles = [
        Patch(facecolor=COLORS["plan"], edgecolor="none", label="Planned"),
        Patch(facecolor=COLORS["completed"], edgecolor="none", label="Completed"),
        Patch(facecolor=COLORS["progress"], edgecolor="none", label="In progress"),
    ]
    ax.legend(handles=handles, loc="upper right")

    for name, date_str in MILESTONES:
        m_date = datetime.fromisoformat(date_str)
        ax.axvline(m_date, color="#d62728", linestyle="--", linewidth=1.2)
        ax.scatter(m_date, -1.0, marker="v", color="#d62728", zorder=5)
        ax.text(m_date, -1.6, f"{name}\n{m_date.strftime('%d %b %Y')}",
                ha="center", va="top", fontsize=9, color="#d62728")

    ax.set_yticks([])
    ax.set_ylim(-2, y + 1)
    ax.set_xlim(datetime(2024, 10, 1), datetime(2025, 6, 15))
    ax.set_xlabel("Timeline (Oct 2024 — Jun 2025)")
    ax.set_title("Midterm Work Plan (proposal-style Gantt)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
