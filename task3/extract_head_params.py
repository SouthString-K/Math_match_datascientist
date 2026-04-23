"""
提取 Task3 模型中两个输出头的参数：
  W_c, b_c（方向分类头）
  W_r, b_r（幅度回归头）
"""
import json
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT_PATH = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm\best_model.pt")
OUTPUT_PATH = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm\head_params.json")


def extract_head_params():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # ckpt 可能是 {"model_state_dict": ...} 或直接是 state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # classifier: Linear(in=64, out=1) → weight (1,64), bias (1,)
    wc = state_dict["classifier.weight"].numpy()   # shape (1, 64)
    bc = state_dict["classifier.bias"].numpy()     # shape (1,)

    # regressor: Linear(in=64, out=1) → weight (1,64), bias (1,)
    wr = state_dict["regressor.weight"].numpy()   # shape (1, 64)
    br = state_dict["regressor.bias"].numpy()     # shape (1,)

    result = {
        "W_c": wc.tolist(),   # [[w1, w2, ..., w64]]
        "b_c": bc.tolist(),   # [bias_c]
        "W_r": wr.tolist(),   # [[w1, w2, ..., w64]]
        "b_r": br.tolist(),   # [bias_r]
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"W_c shape: {wc.shape}  → {wc.tolist()}", flush=True)
    print(f"b_c shape: {bc.shape}  → {bc.tolist()}", flush=True)
    print(f"W_r shape: {wr.shape}  → {wr.tolist()}", flush=True)
    print(f"b_r shape: {br.shape}  → {br.tolist()}", flush=True)
    print(f"\n已保存: {OUTPUT_PATH}", flush=True)

    # 同时打印论文格式
    print("\n" + "=" * 60, flush=True)
    print("论文格式引用：", flush=True)
    print("=" * 60, flush=True)
    print(f"W_c = [{', '.join(f'{w:.6f}' for w in wc[0])}]", flush=True)
    print(f"b_c = {bc[0]:.6f}", flush=True)
    print(f"W_r = [{', '.join(f'{w:.6f}' for w in wr[0])}]", flush=True)
    print(f"b_r = {br[0]:.6f}", flush=True)

    return result


if __name__ == "__main__":
    extract_head_params()
