#!/usr/bin/env python3
"""
Proposal-style Gantt chart with the original timeline (Sept 2025 -- May 2026)
and overlay of completed vs planned segments up to the midterm milestone.
Outputs: artifacts/gantt_midterm_proposal.png
"""
import os
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Proposal tasks (start, end, is_group)
BARS = [
    ("1. Preliminary Work", "2025-09-01", "2025-10-05", True),
    ("1.1 Proposal & scoping", "2025-09-01", "2025-09-21", False),
    ("1.2 Ethics preparation", "2025-09-15", "2025-10-05", False),

    ("2. Literature Review", "2025-09-22", "2025-11-16", True),
    ("2.1 Collaborative inference survey", "2025-09-22", "2025-10-19", False),
    ("2.2 Attention surrogate study", "2025-10-06", "2025-11-02", False),
    ("2.3 Baseline CEM reproduction", "2025-10-20", "2025-11-16", False),

    ("3. Design & Implementation", "2025-11-03", "2026-01-12", True),
    ("3.1 Slot+Cross design", "2025-11-03", "2025-11-23", False),
    ("3.2 Stability tuning & gating", "2025-11-24", "2025-12-14", False),
    ("3.3 Integration & ablations", "2025-12-01", "2026-01-12", False),

    ("4. Interim Report", "2025-12-01", "2025-12-16", True),

    ("5. Evaluation & Improvement", "2026-01-06", "2026-04-05", True),
    ("5.1 Benchmark experiments", "2026-01-06", "2026-02-16", False),
    ("5.2 Robustness analysis", "2026-02-17", "2026-03-15", False),
    ("5.3 Scalability studies", "2026-03-16", "2026-04-05", False),

    ("6. Dissertation & Dissemination", "2026-03-16", "2026-05-14", True),
    ("6.1 Dissertation writing", "2026-03-16", "2026-04-26", False),
    ("6.2 Video script & recording", "2026-04-15", "2026-05-04", False),
    ("6.3 Q&A rehearsal", "2026-05-06", "2026-05-13", False),
]

MILESTONES = [
    ("Interim Report due", "2025-12-16"),
    ("Dissertation due", "2026-04-28"),
]

COLORS = {
    "planned": "#d9d9d9",
    "completed": "#55a868",
    "in_progress": "#dd8452",
}

# Tasks marked as completed irrespective of date (to reflect current status)
COMPLETED_LABELS = {
    "1. Preliminary Work",
    "1.1 Proposal & scoping",
    "1.2 Ethics preparation",
    "2. Literature Review",
    "2.1 Collaborative inference survey",
    "2.2 Attention surrogate study",
    "2.3 Baseline CEM reproduction",
    "4. Interim Report",
    "5.1 Benchmark experiments",
    "5.2 Robustness analysis",
}

# Tasks currently in progress (explicit)
IN_PROGRESS_LABELS = {
    "3. Design & Implementation",
    "3.3 Integration & ablations",
    "5. Evaluation & Improvement",
    "5.3 Scalability studies",
}

# Status cutoff date (midterm checkpoint)
STATUS_DATE = datetime(2025, 12, 20)


def status_color(label: str, end_date: datetime):
    # force-complete specific tasks
    if label in COMPLETED_LABELS:
        return COLORS["completed"]
    if label in IN_PROGRESS_LABELS:
        return COLORS["in_progress"]
    if end_date <= STATUS_DATE:
        return COLORS["completed"]
    return COLORS["planned"]


def main():
    os.environ.setdefault("MPLCONFIGDIR", "./tmp_mpl")
    fig, ax = plt.subplots(figsize=(12, 7))

    y = 0
    y_gap = 0.25
    h_group, h_task = 0.7, 0.5

    for label, start_str, end_str, is_group in reversed(BARS):
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        width = (end_date - start_date).days + 1
        height = h_group if is_group else h_task
        color = status_color(label, end_date)
        ax.barh(y, width, left=start_date, color=color, height=height, edgecolor="none")
        ax.text(start_date, y, label, va="center", ha="left",
                fontsize=9, fontweight="bold" if is_group else "normal")
        y += height + y_gap

    for name, date_str in MILESTONES:
        m_date = datetime.fromisoformat(date_str)
        ax.axvline(m_date, color="#d62728", linestyle="--", linewidth=1.2)
        ax.scatter(m_date, -1.0, marker="v", color="#d62728", zorder=5)
        ax.text(m_date, -1.6, f"{name}\n{m_date.strftime('%d %b %Y')}",
                ha="center", va="top", fontsize=9, color="#d62728")

    ax.set_yticks([])
    ax.set_ylim(-2, y + 1)
    ax.set_xlim(datetime(2025, 9, 1), datetime(2026, 5, 20))
    ax.set_xlabel("Timeline (Sept 2025 — May 2026)")
    ax.set_title("Work Plan Gantt Chart (proposal timeline, interim report status)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(facecolor=COLORS["planned"], edgecolor="none", label="Planned"),
        Patch(facecolor=COLORS["completed"], edgecolor="none", label="Completed"),
        Patch(facecolor=COLORS["in_progress"], edgecolor="none", label="In progress"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/gantt_midterm_proposal.png", dpi=300)
    plt.close()
    print("Saved artifacts/gantt_midterm_proposal.png")


if __name__ == "__main__":
    main()
