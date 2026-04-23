"""
Task 4: Pseudo Three-Week Backtesting
Runner script
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from task4.pipeline import run_backtest, save_results
from task4.schemas import Task4Config


def main() -> int:
    # 配置
    config = Task4Config(
        # Input paths
        events_path=r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results_new.json",
        assoc_matrix_path=r"D:\Math_match\codes\outputs\task2_attribute_correction\assoc_matrix.csv",
        task3_model_path=r"D:\Math_match\codes\outputs\task3\paper_training_lstm\best_model.pt",
        task3_artifacts_path=r"D:\Math_match\codes\outputs\task3\paper_training_lstm\artifacts.pkl",
        company_profiles_path=r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json",
        stock_history_paths=[
            r"D:\Math_match\codes\task3\hs300_history_batch1.json",
            r"D:\Math_match\codes\task3\hs300_history_batch2.json",
        ],
        
        # Output
        output_dir=r"D:\Math_match\codes\outputs\task4",
        
        # Window definitions
        windows=[
            {"window_id": "W1", "dates": ["4.11"], "name": "窗口1"},
            {"window_id": "W2", "dates": ["4.12", "4.13"], "name": "窗口2"},
            {"window_id": "W3", "dates": ["4.14"], "name": "窗口3"},
        ],
        
        # Event selection params
        events_per_window=10,  # 每个窗口取10个事件
        top_events_selected=6,  # 选择TOP 6事件
        priority_types=["政策类", "宏观类", "行业类", "地缘类"],
        
        # Stock selection params
        top_k_companies_per_event=3,  # 每个事件取TOP 3公司
        top_stocks_to_buy=3,  # 买入TOP 3股票
        
        # Trading params
        initial_capital=100000.0,  # 初始资金10万元
    )
    
    print("=" * 80)
    print("Task 4: 伪三周回测实验")
    print("=" * 80)
    print(f"初始资金: {config.initial_capital:,.2f} 元")
    print(f"窗口数量: {len(config.windows)}")
    print(f"每窗口取事件数: {config.events_per_window}")
    print(f"每窗口选择事件数: {config.top_events_selected}")
    print(f"每事件取公司数: {config.top_k_companies_per_event}")
    print(f"每窗口买入股票数: {config.top_stocks_to_buy}")
    print("=" * 80)
    print()
    
    # 运行回测
    summary = run_backtest(config)
    
    # 保存结果
    save_results(summary, Path(config.output_dir))
    
    # 打印总结
    print()
    print("=" * 80)
    print("回测总结")
    print("=" * 80)
    print(f"初始资金: {summary.total_capital:,.2f} 元")
    print(f"最终资金: {summary.final_capital:,.2f} 元")
    print(f"总收益: {summary.total_profit:,.2f} 元")
    print(f"总收益率: {summary.total_return:.2%}")
    print()
    
    for window in summary.windows:
        print(f"{window.window_id} ({window.window_name}):")
        print(f"  日期: {', '.join(window.dates)}")
        print(f"  选择事件数: {len(window.selected_events)}")
        print(f"  股票池大小: {len(window.stock_pool)}")
        print(f"  买入股票数: {len(window.selected_stocks)}")
        print(f"  投资金额: {window.total_investment:,.2f} 元")
        print(f"  收益: {window.total_profit:,.2f} 元")
        print(f"  收益率: {window.total_return:.2%}")
        print()
    
    print("=" * 80)
    print(f"结果已保存到: {config.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
