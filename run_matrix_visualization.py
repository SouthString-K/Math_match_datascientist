import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# 找到可用的中文字体
_font_found = False
for f in fm.fontManager.ttflist:
    if f.name in ("SimHei", "Microsoft YaHei"):
        plt.rcParams["font.sans-serif"] = [f.name]
        plt.rcParams["axes.unicode_minus"] = False
        _font_found = True
        break
if not _font_found:
    plt.rcParams["axes.unicode_minus"] = False

ASSOC_MATRIX_PATH = r"D:\Math_match\codes\outputs\task2_attribute_correction\assoc_matrix.csv"
COMPANY_PROFILES_PATH = r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
EVENTS_PATH = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\task2_matrix_visualization"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import json

    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    # Load data
    matrix = pd.read_csv(ASSOC_MATRIX_PATH, encoding="utf-8-sig", index_col=0)

    profiles_raw = json.loads(Path(COMPANY_PROFILES_PATH).read_text(encoding="utf-8"))
    companies_data = profiles_raw.get("companies", profiles_raw) if isinstance(profiles_raw, dict) else profiles_raw
    code_to_name = {str(c.get("stock_code", "")).strip(): c.get("stock_name", "") for c in companies_data}

    events_raw = json.loads(Path(EVENTS_PATH).read_text(encoding="utf-8"))
    id_to_title = {str(e.get("sample_id", "")).strip(): e.get("title", "")[:18] for e in events_raw}

    # ---- Build the matrix with ellipsis ----
    n_show = 4      # show first n rows/cols
    n_company = matrix.shape[0]   # 300
    n_event = matrix.shape[1]     # 220

    # Show first n_show + ellipsis + last n_show companies/events
    top_n = matrix.index[:n_show].tolist()
    bot_n = matrix.index[-n_show:].tolist()
    left_n = matrix.columns[:n_show].tolist()
    right_n = matrix.columns[-n_show:].tolist()

    # Truncated data blocks
    top_left = matrix.loc[top_n, left_n].values
    top_right = matrix.loc[top_n, right_n].values
    bot_left = matrix.loc[bot_n, left_n].values
    bot_right = matrix.loc[bot_n, right_n].values

    # Company/event labels
    company_labels_top = [code_to_name.get(str(c), str(c)) for c in top_n]
    company_labels_bot = [code_to_name.get(str(c), str(c)) for c in bot_n]
    event_labels_left = [id_to_title.get(str(e), str(e)) for e in left_n]
    event_labels_right = [id_to_title.get(str(e), str(e)) for e in right_n]

    # ---- Assemble full data matrix with ellipsis blocks ----
    h_ellipsis_col = np.full((n_show, 1), np.nan)
    v_ellipsis_row = np.full((1, 2 * n_show + 1), np.nan)

    # Top block
    top_block = np.hstack([top_left, h_ellipsis_col, top_right])
    # Middle block (ellipsis row label area)
    mid_ellipsis = np.full((1, 2 * n_show + 1), np.nan)
    # Bottom block
    bot_block = np.hstack([bot_left, h_ellipsis_col, bot_right])

    data_matrix = np.vstack([top_block, v_ellipsis_row, bot_block])

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    cmap = plt.cm.YlOrRd
    vmin = float(matrix.values.min())
    vmax = float(matrix.values.max())

    im = ax.imshow(data_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Column headers (companies): top row and bottom row of matrix
    all_company_labels = company_labels_top + ["..."] + company_labels_bot
    all_company_codes = top_n + ["..."] + bot_n
    ax.set_xticks(range(len(all_company_labels)))
    ax.set_xticklabels(all_company_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, len(all_company_labels) - 0.5)

    # Row headers (events): left column
    event_row_labels = event_labels_left + ["..."] + event_labels_right
    event_row_codes = left_n + ["..."] + right_n
    ax.set_yticks(range(len(event_row_labels)))
    ax.set_yticklabels(event_row_labels, fontsize=7)
    ax.set_ylim(len(event_row_labels) - 0.5, -0.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Assoc_ij", fontsize=9)

    # Title and labels
    ax.set_title("公司—事件关联矩阵 (Assoc_ij)\n300 家公司 × 220 事件", fontsize=12, pad=10)
    ax.set_xlabel("公司 (Company)", fontsize=9)
    ax.set_ylabel("事件 (Event)", fontsize=9)

    # Add column total header (spanning companies)
    ax2_top = ax.secondary_xaxis("top")
    ax2_top.set_xticks([])
    ax2_top.set_xlabel("← 省略中间省略号 →", fontsize=7, color="gray")

    fig.tight_layout()
    out_path = output / "matrix_structure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] 已保存: {out_path}", flush=True)

    # ---- Also save a text-based representation for reference ----
    txt_lines = []
    txt_lines.append("公司—事件关联矩阵 (Assoc_ij)".center(80, "="))
    txt_lines.append(f"矩阵规模: {n_company} 家公司 × {n_event} 事件\n")

    header = "事件\\公司".ljust(20) + "".join(f"{l:>12}" for l in company_labels_top) + "".join(f"{'>...<':>12}") + "".join(f"{l:>12}" for l in company_labels_bot)
    txt_lines.append(header)
    txt_lines.append("-" * len(header))

    for i, (row_lbl, row_code) in enumerate(zip(event_labels_left, left_n)):
        row_vals = top_left[i]
        line = f"{row_lbl[:18]:<20}" + "".join(f"{v:>12.4f}" for v in row_vals) + "".join(f"{'>...<':>12}") + "".join(f"{v:>12.4f}" for v in top_right[i])
        txt_lines.append(line)

    txt_lines.append(" " * 20 + "..." * 12)
    txt_lines.append(" " * 20 + "..." * 12)

    for i, (row_lbl, row_code) in enumerate(zip(event_labels_right, right_n)):
        row_vals = bot_left[i]
        line = f"{row_lbl[:18]:<20}" + "".join(f"{v:>12.4f}" for v in row_vals) + "".join(f"{'>...<':>12}") + "".join(f"{v:>12.4f}" for v in bot_right[i])
        txt_lines.append(line)

    txt_path = output / "matrix_structure.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"[Viz] 已保存: {txt_path}", flush=True)

    print(f"[Viz] 完成。输出目录: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
