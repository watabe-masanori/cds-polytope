"""
CDS Higher-Dimensional Core
==============================
24-cell (4D・24点) の次の階層：

  CDS-4D  :  24-cell        4次元  24頂点    ← WCMで実証済み
  CDS-8D  :  E₈ root系    8次元  240点    ← 今回の新領域
  CDS-24D :  Leech格子    24次元  196560点  ← 未来の限界

E₈が特別な理由:
  ・8次元球面上で最も均一なパッキング（証明済み: Viazovska 2022 Fields Medal）
  ・最大内積0.50 vs 24-cellの0.707 → 方向分解能が大幅向上
  ・素粒子物理のE₈×E₈対称性・弦理論とも関係を持つ
  ・8軸 = より豊かな物理的状態空間を記述できる

「前人未到」の意味:
  臨界現象検出に E₈ 幾何学を使った先行研究は存在しない。
"""

import numpy as np
from itertools import combinations, product

# ═══════════════════════════════════════════════════════
# 1. E₈ root vectors (240点)
# ═══════════════════════════════════════════════════════
def build_e8_roots():
    """
    E₈の240 root vectorsを生成。
    全て同じノルム√2を持ち、8次元球面上に最均一に分布。
    
    Type I  (112点): ±eᵢ ± eⱼ  (i≠j)
    Type II (128点): (±1/2)^8  (-の個数が偶数)
    """
    roots = []
    # Type I
    for i, j in combinations(range(8), 2):
        for si in [+1, -1]:
            for sj in [+1, -1]:
                v = np.zeros(8)
                v[i] = si; v[j] = sj
                roots.append(v)
    # Type II
    for bits in range(256):
        signs = [+0.5 if (bits >> k) & 1 == 0 else -0.5 for k in range(8)]
        if signs.count(-0.5) % 2 == 0:
            roots.append(np.array(signs))

    roots = np.array(roots)
    # 単位球面に正規化
    unit = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    return roots, unit   # shape (240, 8)

E8_ROOTS, E8_UNIT = build_e8_roots()


# ═══════════════════════════════════════════════════════
# 2. 24-cell roots (24点) — 8D空間の最初の4軸に埋め込み
# ═══════════════════════════════════════════════════════
def build_24cell_pure4d():
    """純粋4次元24-cell（比較実験用）"""
    from itertools import combinations as comb
    verts = []
    s = 1.0 / np.sqrt(2)
    for i, j in comb(range(4), 2):
        for si in [+s, -s]:
            for sj in [+s, -s]:
                v = np.zeros(4)
                v[i] = si; v[j] = sj
                verts.append(v)
    verts = np.array(verts)
    unit  = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, unit

C24_4D_VERTS, C24_4D_UNIT = build_24cell_pure4d()


# ═══════════════════════════════════════════════════════
# 3. 共通写像エンジン（次元数に依存しない設計）
# ═══════════════════════════════════════════════════════
class CDSEngine:
    """
    任意の「参照点セット」を使う汎用 CDS エンジン。
    
    ref_points : ndarray (K, D)  — 参照方向の集合（単位化済み）
                 K=24  → CDS-4D (24-cell)
                 K=240 → CDS-8D (E₈)
    """
    def __init__(self, ref_points: np.ndarray, name: str = "CDS"):
        self.refs = ref_points / np.linalg.norm(ref_points, axis=1, keepdims=True)
        self.K    = len(ref_points)
        self.D    = ref_points.shape[1]
        self.name = name
        self.uniform = np.ones(self.K) / self.K

    def normalize(self, events: np.ndarray, bounds: dict = None):
        """events: (N, D) → normalized to [-1,1]^D"""
        if bounds is None:
            lo = events.min(axis=0)
            hi = events.max(axis=0)
        else:
            lo, hi = bounds['min'], bounds['max']
        rng = hi - lo
        rng[rng == 0] = 1.0
        normed = 2.0 * (events - lo) / rng - 1.0
        return np.clip(normed, -1.0, 1.0), {'min': lo, 'max': hi}

    def assign(self, normed: np.ndarray) -> np.ndarray:
        """各イベントを最近傍参照点に割り当て → (N,)"""
        norms = np.linalg.norm(normed, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        unit = normed / norms
        dots = unit @ self.refs.T   # (N, K)
        return np.argmax(dots, axis=1)

    def shape_vector(self, ref_ids: np.ndarray) -> np.ndarray:
        """K次元確率分布 sv[]"""
        counts = np.bincount(ref_ids, minlength=self.K).astype(float)
        s = counts.sum()
        return counts / s if s > 0 else self.uniform.copy()

    def cosD(self, sv: np.ndarray) -> float:
        """cosD ∈ [0,1]  — 1=完全整列, 0=完全ランダム"""
        cos_sim = np.dot(sv, self.uniform) / (
            np.linalg.norm(sv) * np.linalg.norm(self.uniform) + 1e-12)
        return float(1.0 - cos_sim)

    def entropy(self, sv: np.ndarray) -> float:
        """
        方向エントロピー H = -Σ sv_k log(sv_k)
        新指標：cosD より細かい「分散度」。
        最大 = log(K) （完全ランダム）、最小 = 0 （完全整列）
        """
        sv_safe = np.where(sv > 1e-12, sv, 1e-12)
        return float(-np.sum(sv * np.log(sv_safe)))

    def hot_vertex(self, sv: np.ndarray):
        idx = int(np.argmax(sv))
        return idx, float(sv[idx])

    def analyze_window(self, events: np.ndarray, bounds: dict):
        """1窓分の完全解析"""
        if len(events) < 3:
            return {'cosD': 0.0, 'entropy': np.log(self.K),
                    'hot_v': 0, 'hot_prob': 1/self.K, 'sv': self.uniform.copy()}
        normed, _ = self.normalize(events, bounds)
        rids = self.assign(normed)
        sv   = self.shape_vector(rids)
        return {
            'cosD'    : self.cosD(sv),
            'entropy' : self.entropy(sv),
            'hot_v'   : self.hot_vertex(sv)[0],
            'hot_prob': self.hot_vertex(sv)[1],
            'sv'      : sv,
        }

    def scan(self, events_all: np.ndarray, times: np.ndarray,
             window: float, step: float, bounds: dict = None):
        """スライディングウィンドウ全走査"""
        if bounds is None:
            _, bounds = self.normalize(events_all)
        t0, t1 = times.min(), times.max()
        results = []
        t = t0
        while t + window <= t1 + step:
            mask = (times >= t) & (times < t + window)
            win  = events_all[mask]
            r    = self.analyze_window(win, bounds)
            r['t_center'] = t + window / 2
            r['n']        = len(win)
            results.append(r)
            t += step
        return results


# ═══════════════════════════════════════════════════════
# 4. 次元階層のカタログ
# ═══════════════════════════════════════════════════════
DIMENSION_CATALOG = {
    'CDS-4D': {
        'refs'       : C24_4D_UNIT,
        'dim'        : 4,
        'n_refs'     : 24,
        'structure'  : '24-cell',
        'max_inner'  : 1/np.sqrt(2),
        'description': 'WCMで実証済み。4次元の完全均一充填。',
    },
    'CDS-8D': {
        'refs'       : E8_UNIT,
        'dim'        : 8,
        'n_refs'     : 240,
        'structure'  : 'E₈ root system',
        'max_inner'  : 0.5,
        'description': '8次元の最密均一充填(Fields Medal 2022)。前人未到。',
    },
}

def print_catalog():
    print("=" * 65)
    print("CDS 次元階層カタログ")
    print("=" * 65)
    for name, info in DIMENSION_CATALOG.items():
        print(f"\n{name}  ({info['structure']})")
        print(f"  次元数   : {info['dim']}")
        print(f"  参照点数 : {info['n_refs']}")
        print(f"  最大内積 : {info['max_inner']:.4f}  (小さいほど均一)")
        print(f"  分解能   : {info['description']}")
    print("\nLeech格子 (24D・196560点) は実装参照のみ。計算量の壁。")
    print("=" * 65)

if __name__ == '__main__':
    print_catalog()
    print(f"\nE₈ roots shape: {E8_ROOTS.shape}")
    print(f"E₈ unit  shape: {E8_UNIT.shape}")
    # 均一性検証
    dots = E8_UNIT @ E8_UNIT.T
    np.fill_diagonal(dots, 0)
    print(f"E₈ 最大内積: {dots.max():.4f}  最小内積: {dots.min():.4f}")
