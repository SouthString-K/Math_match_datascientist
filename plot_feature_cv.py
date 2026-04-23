import sys
import matplotlib.pyplot as plt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
import numpy as np

matplotlib.use("Agg")

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（已按CV从低到高排序）
features = ["Intensity", "Heat", "Range", "Attribute"]
cv_values = [0.2360, 0.2713, 0.3595, 0.3863]

# 颜色：蓝灰系，Attribute略深
colors = ["#5B7FA3", "#6B9BC3", "#4A7BA7", "#2E5D8C"]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(features))
bars = ax.bar(x, cv_values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

# 柱子顶部数值标签
for bar, val in zip(bars, cv_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.006,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=12)
ax.set_ylabel("变异系数 (CV)", fontsize=12, color="#333333")
ax.set_ylim(0, 0.46)
ax.tick_params(axis="y", labelsize=10, colors="#555555")
ax.tick_params(axis="x", length=0)

# 网格线
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax.set_axisbelow(True)

# 去除边框
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

# 结论说明文字
fig.text(
    0.5, 0.13,
    "CV 越高，说明该特征对不同事件的相对区分能力越强。",
    ha="center",
    fontsize=10,
    color="#444444",
    style="italic",
)

# 图注
fig.text(
    0.5, 0.06,
    "注：四项特征的典型事件一致性检验均通过，整体一致性为 100%（14/14）。",
    ha="center",
    fontsize=9.5,
    color="#666666",
)

# 标题 & 副标题
fig.suptitle(
    "四项事件量化特征的相对区分能力比较图",
    fontsize=14,
    fontweight="bold",
    color="#1a1a1a",
    y=0.98,
)
ax.set_title(
    "基于变异系数（CV）与典型事件一致性检验的间接验证",
    fontsize=11,
    color="#444444",
    pad=14,
)

plt.tight_layout(rect=[0, 0.19, 1, 0.95])

out_path = r"D:\Math_match\codes\outputs\task1_feature_validation\feature_cv_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[Val] 已保存: {out_path}")
