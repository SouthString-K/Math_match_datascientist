import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.use("Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for f in fm.fontManager.ttflist:
    if f.name in ("SimHei", "Microsoft YaHei"):
        plt.rcParams["font.sans-serif"] = [f.name]
        plt.rcParams["axes.unicode_minus"] = False
        break

ASSOC_MATRIX_PATH = r"D:\Math_match\codes\outputs\task2_attribute_correction\assoc_matrix.csv"
COMPANY_PROFILES_PATH = r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
EVENTS_PATH = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\task2_matrix_visualization"

import json

matrix = pd.read_csv(ASSOC_MATRIX_PATH, encoding="utf-8-sig", index_col=0)

profiles_raw = json.loads(Path(COMPANY_PROFILES_PATH).read_text(encoding="utf-8"))
companies_data = profiles_raw.get("companies", profiles_raw) if isinstance(profiles_raw, dict) else profiles_raw
code_to_name = {str(c.get("stock_code", "")).strip(): c.get("stock_name", "") for c in companies_data}

events_raw = json.loads(Path(EVENTS_PATH).read_text(encoding="utf-8"))
id_to_title = {str(e.get("sample_id", "")).strip(): e.get("title", "")[:18] for e in events_raw}

# Top-15 by mean Assoc
TOPK = 15
company_mean = matrix.mean(axis=1).sort_values(ascending=False)
event_mean = matrix.mean(axis=0).sort_values(ascending=False)

top15_companies = company_mean.index[:TOPK].tolist()
top15_events = event_mean.index[:TOPK].tolist()

sub = matrix.loc[top15_companies, top15_events]

display_index = [f"{code_to_name.get(str(c), str(c))} ({c})" for c in sub.index]
display_cols = [f"{id_to_title.get(str(e), str(e))} ({e})" for e in sub.columns]

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(sub.values.clip(0, 1.5), cmap="Blues", interpolation="nearest", aspect="auto")

ax.set_xticks(range(len(display_cols)))
ax.set_xticklabels(display_cols, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(display_index)))
ax.set_yticklabels(display_index, fontsize=7.5)
ax.set_ylim(len(display_index) - 0.5, -0.5)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Assoc_ij", fontsize=10)

ax.set_title(
    "Top-15 公司 × Top-15 事件 关联强度热力图",
    fontsize=13, fontweight="bold", color="#1a1a1a", pad=12,
)
ax.set_xlabel("事件 (Event)", fontsize=10)
ax.set_ylabel("公司 (Company)", fontsize=10)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out_path = Path(OUTPUT_DIR) / "heatmap_top15x15.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[Viz] Saved: {out_path}")
