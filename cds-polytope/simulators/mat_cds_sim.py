"""
MAT-CDS Simulation Data Generator
===================================
物理的に正確な3フェーズ AE（アコースティックエミッション）データを生成する。

Phase 1: 正常フェーズ   — AEは空間的にランダム分散、低エネルギー
Phase 2: 前兆フェーズ   — AEが主亀裂面方向に空間的整列を始める（スロースリップ相当）
Phase 3: 破断直前フェーズ — AEが特定方向に強く収束、エネルギー急上昇

"""

import numpy as np
import pandas as pd

def generate_ae_data(
    seed: int = 42,
    # 試験体サイズ (mm)
    specimen_x: float = 100.0,
    specimen_y: float = 50.0,
    specimen_z: float = 20.0,
    # フェーズ設定
    phase1_days: float = 80.0,   # 正常フェーズ期間
    phase2_days: float = 60.0,   # 前兆フェーズ期間
    phase3_days: float = 20.0,   # 破断直前フェーズ期間
    # AE発生レート（イベント/日）
    rate1: float = 20.0,
    rate2: float = 60.0,
    rate3: float = 200.0,
    # 主亀裂面の向き（WCMでいうプレートの方向）
    crack_normal: np.ndarray = None,
) -> pd.DataFrame:
    """
    Returns DataFrame columns:
        time (days), x (mm), y (mm), z (mm), energy (aJ), log_energy,
        phase (1/2/3), label
    """
    rng = np.random.default_rng(seed)

    # 主亀裂面の法線ベクトル（単位ベクトル）
    # default: xz面に45°傾いた面 → (1/√2, 0, 1/√2)方向
    if crack_normal is None:
        crack_normal = np.array([1/np.sqrt(2), 0.0, 1/np.sqrt(2)])
    crack_normal = crack_normal / np.linalg.norm(crack_normal)

    rows = []
    total_days = phase1_days + phase2_days + phase3_days

    def poisson_times(rate, duration, t_offset):
        n = rng.poisson(rate * duration)
        return rng.uniform(0, duration, n) + t_offset

    # ── Phase 1: 正常フェーズ ──────────────────────────────────
    t1s = poisson_times(rate1, phase1_days, 0.0)
    for t in t1s:
        # 位置：試験体全体に一様ランダム
        x = rng.uniform(0, specimen_x)
        y = rng.uniform(0, specimen_y)
        z = rng.uniform(0, specimen_z)
        # エネルギー：低い（1〜50 aJ）
        energy = rng.exponential(10.0) + 1.0
        rows.append((t, x, y, z, energy, 1, 'normal'))

    # ── Phase 2: 前兆フェーズ ────────────────────────────────
    t2s = poisson_times(rate2, phase2_days, phase1_days)
    for t in t2s:
        # 位置：徐々に主亀裂面方向に収束
        # 亀裂面中心を specimen 中央に設定
        cx, cy, cz = specimen_x/2, specimen_y/2, specimen_z/2
        # 面内拡散（大）＋面法線方向拡散（小）
        spread_in_plane  = 20.0   # 亀裂面内の拡がり
        spread_normal    = 8.0    # 法線方向の拡がり（前兆：まだ広い）

        # 亀裂面の基底ベクトルを求める
        ref = np.array([0, 1, 0]) if abs(crack_normal[1]) < 0.9 else np.array([1, 0, 0])
        t1_basis = np.cross(crack_normal, ref); t1_basis /= np.linalg.norm(t1_basis)
        t2_basis = np.cross(crack_normal, t1_basis)

        p1 = rng.normal(0, spread_in_plane)
        p2 = rng.normal(0, spread_in_plane)
        pn = rng.normal(0, spread_normal)
        pos = np.array([cx, cy, cz]) + p1*t1_basis + p2*t2_basis + pn*crack_normal
        pos = np.clip(pos, [0,0,0], [specimen_x, specimen_y, specimen_z])

        # エネルギー：中程度（5〜200 aJ）、少し上昇
        energy = rng.exponential(30.0) + 5.0
        rows.append((t, pos[0], pos[1], pos[2], energy, 2, 'precursor'))

    # ── Phase 3: 破断直前フェーズ ─────────────────────────────
    t3s = poisson_times(rate3, phase3_days, phase1_days + phase2_days)
    for t in t3s:
        cx, cy, cz = specimen_x/2, specimen_y/2, specimen_z/2
        spread_in_plane  = 25.0   # 亀裂面内：成長で拡大
        spread_normal    = 2.0    # 法線方向：極めて薄い（収束！）

        ref = np.array([0, 1, 0]) if abs(crack_normal[1]) < 0.9 else np.array([1, 0, 0])
        t1_basis = np.cross(crack_normal, ref); t1_basis /= np.linalg.norm(t1_basis)
        t2_basis = np.cross(crack_normal, t1_basis)

        p1 = rng.normal(0, spread_in_plane)
        p2 = rng.normal(0, spread_in_plane)
        pn = rng.normal(0, spread_normal)   # ← 極めて小さい！
        pos = np.array([cx, cy, cz]) + p1*t1_basis + p2*t2_basis + pn*crack_normal
        pos = np.clip(pos, [0,0,0], [specimen_x, specimen_y, specimen_z])

        # エネルギー：高い（50〜5000 aJ）、急上昇
        energy = rng.exponential(500.0) + 50.0
        rows.append((t, pos[0], pos[1], pos[2], energy, 3, 'pre-fracture'))

    df = pd.DataFrame(rows, columns=['time','x','y','z','energy','phase','label'])
    df = df.sort_values('time').reset_index(drop=True)
    df['log_energy'] = np.log10(df['energy'])
    return df


if __name__ == '__main__':
    df = generate_ae_data()
    print("=== 生成AEデータ概要 ===")
    print(f"総イベント数: {len(df)}")
    print()
    for ph, label in [(1,'normal'),(2,'precursor'),(3,'pre-fracture')]:
        sub = df[df['phase']==ph]
        print(f"Phase {ph} ({label}):")
        print(f"  イベント数 : {len(sub)}")
        print(f"  時間範囲   : {sub['time'].min():.1f} 〜 {sub['time'].max():.1f} 日")
        print(f"  energy平均 : {sub['energy'].mean():.1f} aJ")
        print(f"  x分散      : {sub['x'].std():.2f} mm")
        print(f"  z分散      : {sub['z'].std():.2f} mm")
    print()
    print(df.head())
