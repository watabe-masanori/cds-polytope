"""
FIN-CDS Simulation Data Generator
====================================
株式市場の「震源雲」を生成する。各銘柄の日次状態が1イベント。

4D observation space:
    axis 0 : return_dev    リターン偏差  (ri - r_market) / sigma_r
    axis 1 : vol_dev       ボラティリティ偏差  (vi - v_market) / sigma_v
    axis 2 : corr_break    相関破綻度  1 - rho_i_market
    axis 3 : liq_dev       流動性偏差  (liq_i - liq_market) / sigma_liq

Phase 1 (Bull市場)    : 銘柄がバラバラに動く（ランダム分散）
Phase 2 (不安定期)    : 相関が増大しボラティリティ上昇（前兆）
Phase 3 (クラッシュ前): 全銘柄が同じ方向へ強く収束（整列）
"""

import numpy as np
import pandas as pd

def generate_fin_data(
    seed: int = 42,
    n_stocks: int = 100,
    phase1_days: int = 80,
    phase2_days: int = 60,
    phase3_days: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # ── Phase 1: Bull市場 ──────────────────────────────────
    for day in range(phase1_days):
        t = float(day)
        for _ in range(n_stocks):
            ret_dev  = rng.normal(0.0,  1.0)    # 完全ランダム
            vol_dev  = rng.normal(0.0,  1.0)
            corr_brk = rng.uniform(0.0, 1.0)   # 相関はランダム
            liq_dev  = rng.normal(0.0,  1.0)
            rows.append((t, ret_dev, vol_dev, corr_brk, liq_dev, 1, 'bull'))

    # ── Phase 2: 不安定期（前兆）─────────────────────────
    for day in range(phase2_days):
        t = float(phase1_days + day)
        prog = day / phase2_days   # 0→1で徐々に整列

        # 市場全体の共通因子が増大する
        common_ret = rng.normal(-0.5 * prog, 0.5)   # 徐々に下落方向へ
        common_vol = 1.0 + 2.0 * prog               # ボラティリティ上昇

        for _ in range(n_stocks):
            # 個別ノイズ + 共通因子（比率がprogで変化）
            idio = rng.normal(0.0, 1.0 - 0.6 * prog)
            ret_dev  = common_ret * prog + idio
            vol_dev  = rng.normal(common_vol, 0.5)
            corr_brk = rng.uniform(0.0, 1.0 - 0.7 * prog)  # 相関が高まる
            liq_dev  = rng.normal(-prog, 0.5)               # 流動性低下
            rows.append((t, ret_dev, vol_dev, corr_brk, liq_dev, 2, 'unstable'))

    # ── Phase 3: クラッシュ直前──────────────────────────
    for day in range(phase3_days):
        t = float(phase1_days + phase2_days + day)

        crash_ret = rng.normal(-3.0, 0.3)   # 強い下落方向
        crash_vol = rng.normal( 5.0, 0.5)   # 極大ボラティリティ

        for _ in range(n_stocks):
            ret_dev  = crash_ret + rng.normal(0, 0.2)   # ほぼ全員同じ方向
            vol_dev  = crash_vol + rng.normal(0, 0.3)
            corr_brk = rng.uniform(0.0, 0.1)            # 全銘柄が市場と相関→0
            liq_dev  = rng.normal(-4.0, 0.5)            # 流動性枯渇
            rows.append((t, ret_dev, vol_dev, corr_brk, liq_dev, 3, 'pre-crash'))

    df = pd.DataFrame(rows,
        columns=['time','return_dev','vol_dev','corr_break','liq_dev','phase','label'])
    return df.sort_values('time').reset_index(drop=True)


if __name__ == '__main__':
    df = generate_fin_data()
    print("=== FIN-CDS シミュレーションデータ ===")
    for ph, lbl in [(1,'bull'),(2,'unstable'),(3,'pre-crash')]:
        s = df[df['phase']==ph]
        print(f"Phase {ph} ({lbl}): {len(s)} events, "
              f"ret_dev={s['return_dev'].mean():.2f}, "
              f"vol_dev={s['vol_dev'].mean():.2f}")
