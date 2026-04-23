"""
绘制选股权重分配表（中文版，精美风格）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据 ----------
windows = [
    {
        "title": "窗口1",
        "period": "2025/12/09 → 12/12",
        "stocks": [
            {"code": "000006", "name": "深振业A",  "weight": 0.3781, "buy": 10.16, "sell": 10.01, "shares": 3721, "cost": 37805.36, "revenue": 37247.21, "profit": -558.15},
            {"code": "000002", "name": "万科A",    "weight": 0.3214, "buy": 4.89,  "sell": 5.02,  "shares": 6572, "cost": 32137.08, "revenue": 32991.44, "profit": 854.36},
            {"code": "601328", "name": "交通银行", "weight": 0.3005, "buy": 7.49,  "sell": 7.35,  "shares": 4012, "cost": 30049.88, "revenue": 29488.20, "profit": -561.68},
        ],
        "net_profit": -265.47,
        "net_pct": -0.27,
    },
    {
        "title": "窗口2",
        "period": "2025/12/16 → 12/19",
        "stocks": [
            {"code": "000006", "name": "深振业A",  "weight": 0.3418, "buy": 9.80,  "sell": 9.78,  "shares": 3478, "cost": 34084.40, "revenue": 34014.84, "profit": -69.56},
            {"code": "601328", "name": "交通银行", "weight": 0.3320, "buy": 7.32,  "sell": 7.43,  "shares": 4523, "cost": 33108.36, "revenue": 33605.89, "profit": 497.53},
            {"code": "600926", "name": "杭州银行", "weight": 0.3262, "buy": 15.21, "sell": 15.66, "shares": 2138, "cost": 32518.98, "revenue": 33481.08, "profit": 962.10},
        ],
        "net_profit": 1390.07,
        "net_pct": 1.39,
    },
    {
        "title": "窗口3",
        "period": "2025/12/23 → 12/26",
        "stocks": [
            {"code": "000402", "name": "金融街",   "weight": 0.4226, "buy": 2.78,  "sell": 2.77,  "shares": 15372, "cost": 42734.16, "revenue": 42580.44, "profit": -153.72},
            {"code": "002025", "name": "航天彩虹", "weight": 0.3006, "buy": 47.00, "sell": 49.16, "shares": 646,   "cost": 30362.00, "revenue": 31757.36, "profit": 1395.36},
            {"code": "601077", "name": "渝农商行", "weight": 0.2768, "buy": 6.36,  "sell": 6.34,  "shares": 4401,  "cost": 27990.36, "revenue": 27902.34, "profit": -88.02},
        ],
        "net_profit": 1153.62,
        "net_pct": 1.14,
    },
]

# ---------- 配色 ----------
BG_COLOR      = "#FAFBFD"
HEADER_BG     = "#2B4C7E"
HEADER_FG     = "#FFFFFF"
ROW_EVEN      = "#F0F4FA"
ROW_ODD       = "#FFFFFF"
BORDER_COLOR  = "#C8D1DC"
WIN_BORDER    = "#2B4C7E"
PROFIT_POS    = "#D64045"
PROFIT_NEG    = "#2E8B57"

# ---------- 绘图 ----------
fig = plt.figure(figsize=(20, 9), facecolor=BG_COLOR)
fig.suptitle("选股权重分配与持仓收益明细", fontsize=20, fontweight="bold", color="#1A1A2E", y=0.97)

gs = gridspec.GridSpec(1, 3, wspace=0.12, left=0.03, right=0.97, top=0.88, bottom=0.06)

for col, w in enumerate(windows):
    ax = fig.add_subplot(gs[col])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 窗口标题区域
    title_rect = plt.Rectangle((0.1, 9.0), 9.8, 0.9, facecolor=HEADER_BG, edgecolor="none",
                                zorder=2, clip_on=False)
    ax.add_patch(title_rect)
    ax.text(5, 9.55, w["title"], fontsize=16, fontweight="bold", color=HEADER_FG,
            ha="center", va="center", zorder=3)
    ax.text(5, 9.15, w["period"], fontsize=10, color="#B0C4DE",
            ha="center", va="center", zorder=3)

    # 表头
    headers = ["股票", "权重", "买入", "卖出", "股数", "成本", "卖收", "盈亏"]
    col_x   = [0.6, 2.0, 3.1, 4.1, 5.1, 6.1, 7.3, 8.7]
    col_w   = [1.6, 1.1, 0.95, 0.95, 0.95, 1.15, 1.15, 1.2]

    header_y = 8.4
    header_h = 0.5
    for i, (hdr, cx, cw) in enumerate(zip(headers, col_x, col_w)):
        rect = plt.Rectangle((cx - cw/2, header_y), cw, header_h,
                              facecolor="#3A6BA5", edgecolor=BORDER_COLOR, linewidth=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(cx, header_y + header_h/2, hdr, fontsize=9, fontweight="bold",
                color=HEADER_FG, ha="center", va="center", zorder=3)

    # 数据行
    row_h = 0.65
    for row_idx, stock in enumerate(w["stocks"]):
        y = header_y - (row_idx + 1) * row_h
        bg = ROW_EVEN if row_idx % 2 == 0 else ROW_ODD

        # 行背景
        row_rect = plt.Rectangle((0.1, y), 9.8, row_h, facecolor=bg,
                                  edgecolor=BORDER_COLOR, linewidth=0.3, zorder=1)
        ax.add_patch(row_rect)

        # 股票名 + 代码
        name_str = f"{stock['name']}\n{stock['code']}"
        ax.text(col_x[0], y + row_h/2, name_str, fontsize=8.5, color="#1A1A2E",
                ha="center", va="center", zorder=3, linespacing=1.2)

        # 数值列
        vals = [
            f"{stock['weight']:.2%}",
            f"{stock['buy']:.2f}",
            f"{stock['sell']:.2f}",
            f"{stock['shares']:,}",
            f"{stock['cost']:,.0f}",
            f"{stock['revenue']:,.0f}",
        ]
        for vi, (val, cx) in enumerate(zip(vals, col_x[1:7])):
            ax.text(cx, y + row_h/2, val, fontsize=8.5, color="#333",
                    ha="center", va="center", zorder=3)

        # 盈亏列（红/绿）
        profit = stock["profit"]
        profit_color = PROFIT_POS if profit >= 0 else PROFIT_NEG
        sign = "+" if profit >= 0 else ""
        ax.text(col_x[7], y + row_h/2, f"{sign}{profit:,.0f}",
                fontsize=9, fontweight="bold", color=profit_color,
                ha="center", va="center", zorder=3)

        # 盈亏小条（可视化）
        bar_max_w = 0.9
        max_abs = max(abs(s["profit"]) for s in w["stocks"]) or 1
        bar_w = abs(profit) / max_abs * bar_max_w
        bar_x_start = col_x[7] + 0.6
        bar_y_center = y + row_h/2
        if profit >= 0:
            bar_rect = plt.Rectangle((bar_x_start, bar_y_center - 0.08), bar_w, 0.16,
                                      facecolor=PROFIT_POS, alpha=0.3, zorder=2)
        else:
            bar_rect = plt.Rectangle((bar_x_start, bar_y_center - 0.08), bar_w, 0.16,
                                      facecolor=PROFIT_NEG, alpha=0.3, zorder=2)
        ax.add_patch(bar_rect)

    # 合计行
    summary_y = header_y - 4 * row_h - 0.15
    summary_h = 0.7

    net = w["net_profit"]
    pct = w["net_pct"]
    net_color = PROFIT_POS if net >= 0 else PROFIT_NEG
    sign = "+" if net >= 0 else ""

    summary_rect = plt.Rectangle((0.1, summary_y), 9.8, summary_h,
                                  facecolor="#E8EDF5", edgecolor=WIN_BORDER,
                                  linewidth=1.5, zorder=2, clip_on=False)
    ax.add_patch(summary_rect)

    ax.text(2.5, summary_y + summary_h/2, "窗口净收益", fontsize=10, fontweight="bold",
            color="#1A1A2E", ha="center", va="center", zorder=3)
    ax.text(7.0, summary_y + summary_h/2, f"{sign}{net:,.2f} 元  ({sign}{pct:.2f}%)",
            fontsize=11, fontweight="bold", color=net_color,
            ha="center", va="center", zorder=3)

    # 窗口边框
    border_rect = plt.Rectangle((0.05, summary_y - 0.1), 9.9, 9.9 - summary_y + 0.2,
                                 facecolor="none", edgecolor=WIN_BORDER,
                                 linewidth=2, zorder=4, clip_on=False)
    ax.add_patch(border_rect)

out_path = r"D:\Math_match\codes\outputs\task4\weight_allocation_table.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close("all")
print(f"Saved: {out_path}")
