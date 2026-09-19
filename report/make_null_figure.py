"""Corrected null-model figure: z-scores of observed metrics vs random-graph nulls."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a1a", "#6b6b6b"
LABELS = {"transitivity": "Transitivity", "avg_clustering": "Avg. clustering",
          "avg_path_length": "Avg. path length", "diameter": "Diameter",
          "modularity": "Modularity"}
ORDER = ["transitivity", "avg_clustering", "modularity", "avg_path_length", "diameter"]

resp = pd.read_csv("respondent_null_model.csv").set_index("metric")
stmt = pd.read_csv("null_model_corrected.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=False)

def style(ax, rows, title):
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    for v in (-1.96, 1.96):
        ax.axvline(v, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[m] for m in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("z-score vs null (dashed = ±1.96)", color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

# Panel A: respondent network, configuration null only (one series -> no legend)
rows_a = [m for m in ORDER if m in resp.index]
za = [resp.loc[m, "z"] for m in rows_a]
axes[0].barh(range(len(rows_a)), za, color=BLUE, height=0.45, zorder=3)
for i, (m, z) in enumerate(zip(rows_a, za)):
    p = resp.loc[m, "p_two_sided"]
    # negative bars: label to the RIGHT of zero so it never collides with the y-tick
    x = z + 0.6 if z >= 0 else 0.6
    axes[0].text(x, i, f"z={z:.1f}  p={p:.3f}", va="center",
                 ha="left", fontsize=8.5, color=INK)
axes[0].set_xlim(-6, 32)
style(axes[0], rows_a, "A. Respondent network (96 nodes)\nvs degree-preserving null")

# Panel B: statement network, two nulls
rows_b = ORDER
w = 0.36
for k, (name, color) in enumerate([("Configuration", BLUE), ("ER", ORANGE)]):
    sub = stmt[stmt["null"] == name].set_index("metric")
    zb = [sub.loc[m, "z"] for m in rows_b]
    axes[1].barh(np.arange(len(rows_b)) + (k - 0.5) * w, zb, height=w * 0.92,
                 color=color, label=f"{name} null", zorder=3)
    for i, (m, z) in enumerate(zip(rows_b, zb)):
        off = 0.12 if z >= 0 else -0.12
        axes[1].text(z + off, i + (k - 0.5) * w, f"{z:.1f}", va="center",
                     ha="left" if z >= 0 else "right", fontsize=8, color=INK)
axes[1].set_xlim(-2.6, 5.6)
style(axes[1], rows_b, "B. Statement network (60 nodes)\nvs ER and degree-preserving nulls")
axes[1].legend(frameon=False, loc="lower right", fontsize=9)

fig.suptitle("Observed network structure vs random-graph nulls (500 replicates, unweighted topology)",
             fontsize=12, color=INK, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig2_null_models.png", dpi=200, bbox_inches="tight", facecolor="white")
print("written")
