"""Variance vs discrimination, facetted by domain.

Faceting (not colour alone) carries domain identity, so the four categorical
hues are never asked to separate marks inside one panel.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HUE = {"Technology": "#2a78d6", "Education": "#eb6834",
       "Ethics/Society": "#1baf7a", "Environment": "#eda100"}
INK, MUTED, GHOST = "#1a1a1a", "#6b6b6b", "#d8d8d8"
LABEL = {"T12": (8, 6), "T13": (8, -4), "E03": (8, -2), "E09": (8, 4), "E04": (-8, 8)}

r = pd.read_csv("variance_vs_discrimination.csv")
r["code"] = r["statement"].str[:3]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True)
for ax, dom in zip(axes.ravel(), ["Technology", "Education", "Ethics/Society", "Environment"]):
    sub = r[r.domain == dom]
    ax.scatter(r["std"], r["eta_sq"], s=26, color=GHOST, zorder=2, linewidths=0)
    ax.scatter(sub["std"], sub["eta_sq"], s=52, color=HUE[dom], zorder=3,
               edgecolors="white", linewidths=1.2)
    ax.set_title(f"{dom}   (mean η² = {sub['eta_sq'].mean():.3f})",
                 fontsize=10.5, color=INK, loc="left")
    for code, off in LABEL.items():
        row = sub[sub.code == code]
        if not row.empty:
            ax.annotate(code, (row["std"].iloc[0], row["eta_sq"].iloc[0]),
                        textcoords="offset points", xytext=off, fontsize=9,
                        color=INK, fontweight="bold")
    ax.grid(color="#ededed", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")

for ax in axes[1]:
    ax.set_xlabel("Response standard deviation  (how much the class disagrees)", color=MUTED, fontsize=9.5)
for ax in axes[:, 0]:
    ax.set_ylabel("η²  (how much that disagreement\naligns with community structure)", color=MUTED, fontsize=9.5)

fig.suptitle("Disagreement is not the same as division: variance vs community discrimination, per statement",
             fontsize=12.5, color=INK, y=0.985)
fig.text(0.5, 0.945, "Grey = all 60 statements, shown in every panel for context; coloured = that domain's 15.",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.935])
fig.savefig("figures/fig4_variance_vs_discrimination.png", dpi=200, bbox_inches="tight", facecolor="white")
print("written")
