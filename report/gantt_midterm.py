import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------------------------------------------------------
# Gantt for midterm report
# Planned windows come from the proposal; actual/progress updated to current.
# Colors: planned (light gray), completed (green), in-progress (orange).
# -----------------------------------------------------------------------------

Plan = {
    "Literature review & proposal": (dt.date(2024, 10, 1), dt.date(2024, 11, 30)),
    "Baseline reproduction (GMM)": (dt.date(2024, 12, 1), dt.date(2024, 12, 31)),
    "Gated-attention surrogate": (dt.date(2025, 1, 1), dt.date(2025, 1, 31)),
    "Slot + gated x-attn surrogate": (dt.date(2025, 2, 1), dt.date(2025, 2, 28)),
    "CIFAR-10 experiments & analysis": (dt.date(2025, 2, 1), dt.date(2025, 3, 31)),
    "Fusion design & ablations": (dt.date(2025, 3, 15), dt.date(2025, 4, 15)),
    "Extended evaluation (cuts/datasets)": (dt.date(2025, 4, 15), dt.date(2025, 4, 30)),
    "Final write-up & polish": (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
}

# Adjust these dates to reflect current progress
Actual = {
    "Literature review & proposal": (dt.date(2024, 10, 1), dt.date(2024, 11, 30)),
    "Baseline reproduction (GMM)": (dt.date(2024, 12, 1), dt.date(2024, 12, 31)),
    "Gated-attention surrogate": (dt.date(2025, 1, 1), dt.date(2025, 1, 31)),
    "Slot + gated x-attn surrogate": (dt.date(2025, 2, 1), dt.date(2025, 2, 28)),
    "CIFAR-10 experiments & analysis": (dt.date(2025, 2, 1), dt.date(2025, 3, 20)),  # ongoing
    "Fusion design & ablations": (dt.date(2025, 3, 20), dt.date(2025, 4, 15)),        # scheduled/ongoing
    "Extended evaluation (cuts/datasets)": (dt.date(2025, 4, 20), dt.date(2025, 4, 30)),
    "Final write-up & polish": (dt.date(2025, 5, 1), dt.date(2025, 5, 31)),
}

def draw_gantt(outfile="gantt_midterm.png"):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    y0, h, gap = 10, 6, 6
    yticks, ylabels = [], []

    for i, task in enumerate(Plan.keys()):
        y = y0 + i * (h + gap)
        yticks.append(y + h / 2)
        ylabels.append(task)

        p_start, p_end = Plan[task]
        ax.broken_barh(
            [(mdates.date2num(p_start), (p_end - p_start).days)],
            (y, h),
            facecolors="#d9d9d9",
            edgecolor="#a0a0a0",
            label="Planned" if i == 0 else None,
            alpha=0.9,
            linewidth=0.8,
        )

        a_start, a_end = Actual[task]
        status_color = "#55a868" if a_end <= dt.date.today() else "#dd8452"
        status_label = "Completed" if status_color == "#55a868" else "In progress"
        ax.broken_barh(
            [(mdates.date2num(a_start), (a_end - a_start).days)],
            (y, h),
            facecolors=status_color,
            edgecolor="#555555",
            label=status_label if i == 0 else None,
            alpha=0.95,
            linewidth=0.8,
        )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(
        mdates.date2num(dt.date(2024, 10, 1)),
        mdates.date2num(dt.date(2025, 6, 1)),
    )
    ax.set_xlabel("2024–2025 timeline")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    draw_gantt("gantt_midterm.png")
