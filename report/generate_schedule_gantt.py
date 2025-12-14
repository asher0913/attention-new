#!/usr/bin/env python3
"""
Generate Gantt chart for Slot+Cross Attention CEM project timeline.
Outputs: artifacts/gantt_slot_cross_attention_cem_schedule.png
"""

import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 项目阶段与子任务（起止日期均为 ISO 格式）
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
    ("4.1 Interim report drafting", "2025-12-01", "2025-12-12", False),

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

def main():
    os.environ.setdefault("MPLCONFIGDIR", "./tmp_mpl")  # 避免权限问题

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (label, start_str, end_str, is_group) in enumerate(BARS):
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        width = (end_date - start_date).days + 1
        y = len(BARS) - 1 - idx
        color = "#1f77b4" if is_group else "#98c1ff"
        height = 0.7 if is_group else 0.5
        ax.barh(y, width, left=start_date, color=color, height=height, edgecolor="none")
        ax.text(start_date, y + (0.25 if is_group else 0.0), label,
                va="center", ha="left", fontsize=9,
                fontweight="bold" if is_group else "normal")

    for name, date_str in MILESTONES:
        date = datetime.fromisoformat(date_str)
        ax.axvline(date, color="#d62728", linestyle="--", linewidth=1.2)
        ax.scatter(date, -1, marker="v", color="#d62728", zorder=5)
        ax.text(date, -1.5,
                f"{name}\n{date.strftime('%d %b %Y')}",
                ha="center", va="top", fontsize=9, color="#d62728")

    ax.set_ylim(-3, len(BARS) + 0.5)
    ax.set_xlim(datetime(2025, 9, 1), datetime(2026, 5, 20))
    ax.set_yticks([])
    ax.set_xlabel("Timeline (Sept 2025 — May 2026)")
    ax.set_title("Work Plan Gantt Chart for Investigation of Defense Mechanisms against Model Inversion Attacks", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/gantt_slot_cross_attention_cem_schedule.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
