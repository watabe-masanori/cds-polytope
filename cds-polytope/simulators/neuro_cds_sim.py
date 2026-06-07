"""
NEURO-CDS Simulation Data Generator
======================================
脳内の神経発火イベントを「震源雲」として生成する。

4D observation space:
    axis 0 : pos_ap     前後軸空間位置（Anterior-Posterior）
    axis 1 : pos_lr     左右軸空間位置（Left-Right）
    axis 2 : freq_band  周波数帯（δ=0 ～ γ=1、浅い/深いの比喩）
    axis 3 : phase_cyc  脳波周期内の発火位相 (0〜2π → normalized)

Phase 1 (正常状態)   : 発火がランダム分散（各チャンネル独立）
Phase 2 (前発作期)   : 特定チャンネル群での同期が始まる（スロースリップ相当）
Phase 3 (発作直前)   : 全体伝播・高周波同期・位相ロック（整列完成）
"""

import numpy as np
import pandas as pd

def generate_neuro_data(
    seed: int = 42,
    n_channels: int = 32,
    # フェーズ秒数（EEGスケール：秒）
    phase1_sec: float = 80.0,
    phase2_sec: float = 60.0,
    phase3_sec: float = 20.0,
    # 発火レート（スパイク/秒）
    rate1: float = 20.0,
    rate2: float = 50.0,
    rate3: float = 150.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # チャンネルの物理的位置（前後・左右）
    ch_ap = rng.uniform(-1, 1, n_channels)   # Anterior-Posterior
    ch_lr = rng.uniform(-1, 1, n_channels)   # Left-Right
    # てんかん焦点：後頭部左（ap=-0.6, lr=-0.5）
    focus_ap, focus_lr = -0.6, -0.5

    rows = []

    def poisson_times(rate, duration, t_offset):
        n = rng.poisson(rate * duration)
        return rng.uniform(0, duration, n) + t_offset

    # ── Phase 1: 正常状態──────────────────────────────
    t1s = poisson_times(rate1, phase1_sec, 0.0)
    for t in t1s:
        ch = rng.integers(0, n_channels)
        pos_ap    = ch_ap[ch] + rng.normal(0, 0.2)
        pos_lr    = ch_lr[ch] + rng.normal(0, 0.2)
        freq_band = rng.uniform(0, 1)          # 全周波数帯にランダム
        phase_cyc = rng.uniform(-1, 1)         # 位相もランダム
        rows.append((t, pos_ap, pos_lr, freq_band, phase_cyc, 1, 'normal'))

    # ── Phase 2: 前発作期（スロースリップ相当）───────────
    t2s = poisson_times(rate2, phase2_sec, phase1_sec)
    for t in t2s:
        prog = (t - phase1_sec) / phase2_sec   # 0→1

        if rng.random() < 0.3 + 0.5 * prog:
            # 焦点付近での同期発火（徐々に増加）
            pos_ap = focus_ap + rng.normal(0, 0.3 * (1 - prog))
            pos_lr = focus_lr + rng.normal(0, 0.3 * (1 - prog))
            freq_band = rng.uniform(0.6, 1.0)       # 高周波が増加（γ帯）
            phase_cyc = rng.normal(0.0, 0.5 * (1 - prog))  # 位相が収束
        else:
            # バックグラウンド発火
            ch = rng.integers(0, n_channels)
            pos_ap    = ch_ap[ch] + rng.normal(0, 0.2)
            pos_lr    = ch_lr[ch] + rng.normal(0, 0.2)
            freq_band = rng.uniform(0, 1)
            phase_cyc = rng.uniform(-1, 1)

        rows.append((t, pos_ap, pos_lr, freq_band, phase_cyc, 2, 'pre-ictal'))

    # ── Phase 3: 発作直前（全体伝播）──────────────────
    t3s = poisson_times(rate3, phase3_sec, phase1_sec + phase2_sec)
    for t in t3s:
        # 全チャンネルが焦点方向に引き寄せられる
        pos_ap    = rng.normal(focus_ap, 0.15)   # 強く収束
        pos_lr    = rng.normal(focus_lr, 0.15)
        freq_band = rng.uniform(0.8, 1.0)        # γ帯支配
        phase_cyc = rng.normal(0.3, 0.1)         # 強い位相ロック
        rows.append((t, pos_ap, pos_lr, freq_band, phase_cyc, 3, 'pre-seizure'))

    df = pd.DataFrame(rows,
        columns=['time','pos_ap','pos_lr','freq_band','phase_cyc','phase','label'])
    return df.sort_values('time').reset_index(drop=True)


if __name__ == '__main__':
    df = generate_neuro_data()
    print("=== NEURO-CDS シミュレーションデータ ===")
    for ph, lbl in [(1,'normal'),(2,'pre-ictal'),(3,'pre-seizure')]:
        s = df[df['phase']==ph]
        print(f"Phase {ph} ({lbl}): {len(s)} events, "
              f"pos_ap={s['pos_ap'].mean():.2f}, "
              f"freq={s['freq_band'].mean():.2f}, "
              f"phase_cyc={s['phase_cyc'].mean():.2f}")
