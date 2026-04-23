"""
累计收益曲线图（仿论文风格）
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

windows     = [0, 1, 2, 3]
capital     = [100000.0, 99734.53, 101124.60, 102278.22]
cum_return  = [(c - 100000.0) / 100000.0 for c in capital]

fig, ax = plt.subplots(figsize=(8, 4))

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

ax.plot(windows, cum_return, "b-o", linewidth=2, markersize=7, label="Portfolio Cum. Return")

ax.fill_between(windows, 0, cum_return, where=[r >= 0 for r in cum_return],
                color="red", alpha=0.15, interpolate=True)
ax.fill_between(windows, 0, cum_return, where=[r < 0 for r in cum_return],
                color="green", alpha=0.15, interpolate=True)

for w, cr in zip(windows[1:], cum_return[1:]):
    ax.annotate(f"{cr:+.2%}",
                xy=(w, cr), xytext=(0, 12),
                textcoords="offset points",
                ha="center", fontsize=9,
                color="red" if cr >= 0 else "green")

ax.set_xlabel("Window", fontsize=11)
ax.set_ylabel("Cumulative Return", fontsize=11)
ax.set_title("Pseudo 3-Week Backtest - Equity Curve", fontsize=13)
ax.set_xticks(windows)
ax.set_xticklabels(["Init", "W1\n12/09-12/12", "W2\n12/16-12/19", "W3\n12/23-12/26"])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

plt.tight_layout()
out_path = r"D:\Math_match\codes\outputs\task4\equity_curve.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
