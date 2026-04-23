import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据（按均值从高到低排列）
features = ["Attribute", "Heat", "Intensity", "Range"]
mins   = [0.225,  0.30,  0.33,  0.30 ]
means  = [0.8627, 0.8250, 0.8044, 0.7705]
maxs   = [1.44,   1.0,   1.0,   1.0  ]

# 颜色
color_minmax = "#7FA3C4"
color_mean   = "#2E5D8C"

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = np.arange(len(features))

# 画区间线
for i, (mn, mx) in enumerate(zip(mins, maxs)):
    ax.plot([mn, mx], [i, i], color=color_minmax, linewidth=2.5, zorder=2)

# min 端点
ax.scatter(mins, y, color=color_minmax, s=70, zorder=3, label="最小值", edgecolors="white", linewidths=0.8)
# max 端点
ax.scatter(maxs, y, color=color_minmax, s=70, zorder=3, label="最大值", marker="|", linewidths=2, edgecolors="white")
# mean 标记（深色圆点，略大）
ax.scatter(means, y, color=color_mean, s=110, zorder=4, label="均值", edgecolors="white", linewidths=1.2, marker="o")

# 数值标注
for i, (mn, mu, mx) in enumerate(zip(mins, means, maxs)):
    # min
    ax.text(mn - 0.025, i, f"{mn:.3f}", ha="right", va="center", fontsize=9.5, color="#444444")
    # mean
    ax.text(mu, i + 0.18, f"{mu:.4f}", ha="center", va="bottom", fontsize=10, color=color_mean, fontweight="bold")
    # max
    ax.text(mx + 0.025, i, f"{mx:.2f}", ha="left", va="center", fontsize=9.5, color="#444444")

# 纵轴
ax.set_yticks(y)
ax.set_yticklabels(features, fontsize=12)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("特征取值", fontsize=12, color="#333333")

# x轴范围
ax.set_xlim(-0.05, 1.65)

# 网格
ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax.set_axisbelow(True)

# 去除边框
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")

# 图例
ax.legend(loc="upper right", fontsize=10, framealpha=0.9, edgecolor="#cccccc")

# 注释文字
fig.text(
    0.5, 0.03,
    "注：特征取值范围合理，说明事件特征能够在不同事件之间形成有效区分。",
    ha="center", fontsize=9.5, color="#555555",
)

# 标题
fig.suptitle(
    "四项事件量化特征的取值范围与均值比较图",
    fontsize=14, fontweight="bold", color="#1a1a1a", y=0.98,
)
ax.set_title(
    "基于描述统计的取值合理性验证（间接验证）",
    fontsize=11, color="#444444", pad=14,
)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])

out_path = r"D:\Math_match\codes\outputs\task1_feature_validation\feature_range_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[Val] Saved: {out_path}")
