"""
MAT-CDS Core Library
=====================
Critical-transition Detection Sensor for Material Fracture
based on 24-cell polytope geometry (WCM architecture)

4D observation space:
    axis 0 (lon equiv) : AE source position x  (normalized)
    axis 1 (lat equiv) : AE source position y  (normalized)
    axis 2 (dep equiv) : AE source position z  (normalized)
    axis 3 (mag equiv) : AE energy (log-scaled, normalized)

Author: Watabe-Claude Method adaptation for Materials Science
"""

import numpy as np
from itertools import combinations

# ─────────────────────────────────────────────
# 1. 24-cell 頂点定義
# ─────────────────────────────────────────────
def build_24cell_vertices():
    """
    24-cell の 24 頂点を生成する。
    座標形式: (±1/√2, ±1/√2, 0, 0) の全軸ペア順列
    → 4C2=6 ペア × 4 符号組 = 24 頂点
    """
    verts = []
    s = 1.0 / np.sqrt(2)
    for i, j in combinations(range(4), 2):
        for si in [+s, -s]:
            for sj in [+s, -s]:
                v = [0.0, 0.0, 0.0, 0.0]
                v[i] = si
                v[j] = sj
                verts.append(v)
    return np.array(verts)  # shape (24, 4)

VERTS = build_24cell_vertices()

def vertex_label(v_idx):
    """頂点番号 → 4D座標の簡潔表示"""
    v = VERTS[v_idx]
    s = 1/np.sqrt(2)
    def fmt(x):
        if   abs(x - s)  < 1e-9: return "+"
        elif abs(x + s)  < 1e-9: return "-"
        else:                     return "0"
    axes = ["x","y","z","e"]
    parts = [f"{axes[i]}{fmt(v[i])}" for i in range(4) if abs(v[i]) > 1e-9]
    return f"V{v_idx:02d}({''.join(parts)})"


# ─────────────────────────────────────────────
# 2. AE イベントの 24-cell 写像
# ─────────────────────────────────────────────
def normalize_events(events, bounds=None):
    """
    events: ndarray shape (N, 4)  [x, y, z, log_energy]
    bounds: dict with 'min' and 'max' arrays shape (4,)
            None → 自動算出
    Returns normalized events in [-1, 1]^4
    """
    if bounds is None:
        lo = events.min(axis=0)
        hi = events.max(axis=0)
    else:
        lo = bounds['min']
        hi = bounds['max']
    rng = hi - lo
    rng[rng == 0] = 1.0  # ゼロ除算防止
    normed = 2.0 * (events - lo) / rng - 1.0
    return np.clip(normed, -1.0, 1.0), {'min': lo, 'max': hi}

def assign_vertices(normed_events):
    """
    各 AE イベントを最近傍 24-cell 頂点に割り当て。
    Returns: vertex_ids shape (N,)
    """
    # コサイン類似度（単位球上での最近傍）
    norms = np.linalg.norm(normed_events, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit_ev = normed_events / norms          # (N, 4)
    dots = unit_ev @ VERTS.T                 # (N, 24)
    return np.argmax(dots, axis=1)           # (N,)


# ─────────────────────────────────────────────
# 3. 形状ベクトル sv[] と整列指標
# ─────────────────────────────────────────────
def compute_shape_vector(vertex_ids, n_verts=24):
    """
    頂点割り当てリスト → 24次元確率分布 sv[]
    """
    counts = np.bincount(vertex_ids, minlength=n_verts).astype(float)
    total = counts.sum()
    if total == 0:
        return np.ones(n_verts) / n_verts
    return counts / total

def compute_cosD(sv):
    """
    cosD = cosine distance between sv[] and uniform distribution
    uniform = [1/24, 1/24, ..., 1/24]
    cosD ∈ [0, 1];  0 = random,  1 = perfectly aligned
    """
    uniform = np.ones(24) / 24.0
    dot = np.dot(sv, uniform)
    norm_sv   = np.linalg.norm(sv)
    norm_uni  = np.linalg.norm(uniform)
    if norm_sv < 1e-12:
        return 0.0
    cos_sim = dot / (norm_sv * norm_uni)
    return float(1.0 - cos_sim)   # 距離 = 1 - 類似度

def hot_idle_vertices(sv, top_n=3):
    """
    hot_v  : 最大占有率の頂点 (FormA 候補)
    idle_v : 最小占有率の頂点
    Returns: (hot_v_idx, hot_v_prob, idle_v_idx, idle_v_prob)
    """
    sorted_idx = np.argsort(sv)[::-1]
    hot  = sorted_idx[:top_n]
    idle = sorted_idx[-top_n:]
    return hot[0], sv[hot[0]], idle[0], sv[idle[0]]


# ─────────────────────────────────────────────
# 4. スライディングウィンドウ処理
# ─────────────────────────────────────────────
def sliding_window_analysis(events_df, window_days, step_days,
                             t_col='time', bounds=None,
                             feat_cols=None):
    """
    events_df : pandas DataFrame  with columns [t, feat0, feat1, feat2, feat3]
    window_days, step_days : float
    bounds    : 正規化基準 (None → 全データから算出)
    feat_cols : list of 4 column names (None → ['x','y','z','log_energy'])
    
    Returns list of dicts:
        t_center, n_events, sv, cosD, hot_v, hot_prob, idle_v, idle_prob
    """
    import pandas as pd

    if feat_cols is None:
        feat_cols = ['x', 'y', 'z', 'log_energy']

    t_arr = events_df[t_col].values
    feat  = events_df[feat_cols].values

    if bounds is None:
        _, bounds = normalize_events(feat)

    t_start = t_arr.min()
    t_end   = t_arr.max()

    results = []
    t = t_start
    while t + window_days <= t_end + step_days:
        mask = (t_arr >= t) & (t_arr < t + window_days)
        win_events = feat[mask]
        n = len(win_events)
        t_center = t + window_days / 2.0

        if n >= 5:
            normed, _ = normalize_events(win_events, bounds)
            vids = assign_vertices(normed)
            sv   = compute_shape_vector(vids)
            cosD = compute_cosD(sv)
            hv, hp, iv, ip = hot_idle_vertices(sv)
        else:
            sv   = np.ones(24)/24
            cosD = 0.0
            hv, hp, iv, ip = 0, 1/24, 0, 1/24

        results.append({
            't_center' : t_center,
            'n_events' : n,
            'sv'       : sv,
            'cosD'     : cosD,
            'hot_v'    : hv,
            'hot_prob' : hp,
            'idle_v'   : iv,
            'idle_prob': ip,
        })
        t += step_days

    return results


# ─────────────────────────────────────────────
# 5. パーミュテーション検定
# ─────────────────────────────────────────────
def permutation_test(vertex_ids, target_vertex, n_perm=1000, seed=42):
    """
    観測された target_vertex の占有率が偶然起きる確率を推定する。
    Returns p_value (float)
    """
    rng = np.random.default_rng(seed)
    obs_rate = np.mean(vertex_ids == target_vertex)
    count_extreme = 0
    n = len(vertex_ids)
    for _ in range(n_perm):
        shuffled = rng.integers(0, 24, size=n)
        null_rate = np.mean(shuffled == target_vertex)
        if null_rate >= obs_rate:
            count_extreme += 1
    return (count_extreme + 1) / (n_perm + 1)


# ─────────────────────────────────────────────
# 6. V転換検出
# ─────────────────────────────────────────────
def detect_v_transitions(window_results, min_hold=3):
    """
    hot_v が連続 min_hold ウィンドウ以上同じ頂点を保持した後に
    変化した場合を「V転換イベント」として記録する。
    
    Returns list of dicts:
        t_transition, from_v, to_v, hold_count, t_hold_start
    """
    transitions = []
    if not window_results:
        return transitions

    cur_v     = window_results[0]['hot_v']
    hold_cnt  = 1
    hold_start = window_results[0]['t_center']

    for wr in window_results[1:]:
        v = wr['hot_v']
        if v == cur_v:
            hold_cnt += 1
        else:
            if hold_cnt >= min_hold:
                transitions.append({
                    't_transition': wr['t_center'],
                    'from_v'      : cur_v,
                    'to_v'        : v,
                    'hold_count'  : hold_cnt,
                    't_hold_start': hold_start,
                })
            cur_v      = v
            hold_cnt   = 1
            hold_start = wr['t_center']

    return transitions


print("mat_cds_core.py  ロード完了")
print(f"24-cell 頂点数: {len(VERTS)}")
print("頂点リスト (V00〜V05):")
for i in range(6):
    print(f"  {vertex_label(i)}  {VERTS[i]}")
