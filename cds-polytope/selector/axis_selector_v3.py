"""
AutoAxisSelector v0.3
=======================
【 v0.3 の核心的改善 】

問題：有意義な軸が n_select 本に満たない場合、
      v0.2 は意味のない軸（rand・outlier）を無理やり詰め込んでいた。

解決：ゼロパディング戦略
    ① 感受性スコアが SENS_THRESHOLD を超える軸を「有効軸」と判定
    ② 有効軸が n_select 未満なら、残りのスロットを 0 で埋める
    ③ ゼロ列は E₈ の写像で「方向なし」として振る舞う
       → 情報のない軸を追加するより、0 の方が S/N が高い

数学的根拠：
    8次元空間でいくつかの成分が常に 0 の場合、
    実質的に低次元部分空間に制限される。
    これは「不完全な 8D」ではなく「clean な kD (k < 8)」として機能する。

【 現実データへの橋渡し 】
    このモジュールは「任意の DataFrame を投入すれば
    自動的に最良軸を選択して CDS 解析を走らせる」
    汎用パイプラインとして設計してある。
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

from cds_highdim_core import CDSEngine, E8_UNIT, C24_4D_UNIT


# ═══════════════════════════════════════════════════════
# AxisDiagnostics v0.3（v0.2 を継承・改良）
# ═══════════════════════════════════════════════════════
class AxisDiagnosticsV3:
    def __init__(self, feat, times, phases,
                 corr_threshold=0.85, sens_threshold=0.30):
        self.feat   = feat
        self.times  = times
        self.phases = phases
        self.N, self.D = feat.shape
        self.corr_threshold = corr_threshold
        self.sens_threshold = sens_threshold
        self._run()

    def _run(self):
        F, ph = self.feat, self.phases

        # 感受性：Cohen's d × MWU 効果量
        self.sensitivity = np.zeros(self.D)
        for d in range(self.D):
            x1 = F[ph==1, d]; x3 = F[ph==3, d]
            if len(x1)<5 or len(x3)<5: continue
            pooled = np.sqrt((x1.std()**2 + x3.std()**2)/2) + 1e-9
            d_cohen = abs(x3.mean() - x1.mean()) / pooled
            try:
                stat,_ = mannwhitneyu(x1, x3, alternative='two-sided')
                r_mwu  = abs(1 - 2*stat/(len(x1)*len(x3)))
            except: r_mwu = 0.0
            self.sensitivity[d] = d_cohen * r_mwu

        # 相関行列
        self.corr_mat = np.eye(self.D)
        for i in range(self.D):
            for j in range(i+1, self.D):
                r,_ = spearmanr(F[:,i], F[:,j])
                self.corr_mat[i,j] = abs(r)
                self.corr_mat[j,i] = abs(r)

        # 非定常性（分散の変動係数）
        self.nonstationarity = np.zeros(self.D)
        bounds = np.linspace(self.times.min(), self.times.max(), 6)
        for d in range(self.D):
            vs = [np.var(F[(self.times>=bounds[k])&
                           (self.times<bounds[k+1]), d])
                  for k in range(5)
                  if ((self.times>=bounds[k])&
                      (self.times<bounds[k+1])).sum()>5]
            if vs:
                self.nonstationarity[d] = np.std(vs)/(np.mean(vs)+1e-9)

        # 外れ値率
        self.outlier_rate = np.zeros(self.D)
        for d in range(self.D):
            mu,sig = F[:,d].mean(), F[:,d].std()
            if sig>0:
                self.outlier_rate[d] = np.mean(
                    np.abs(F[:,d]-mu)/sig>3.5)

    def solo_score(self, d):
        s = self.sensitivity[d]
        return (s
                / (1 + self.nonstationarity[d])
                / (1 + self.outlier_rate[d]*10))

    def score(self, d, selected, power=2):
        mc = (max(self.corr_mat[d,s] for s in selected)
              if selected else 0.0)
        return (self.sensitivity[d]
                * (1-mc)**power
                / (1+self.nonstationarity[d])
                / (1+self.outlier_rate[d]*10))

    def pre_filter(self):
        """高相関ペアの事前排除"""
        candidates = list(range(self.D))
        removed    = {}
        solo = np.array([self.solo_score(d) for d in range(self.D)])
        changed = True
        while changed:
            changed = False
            for i in candidates:
                for j in candidates:
                    if j<=i: continue
                    if self.corr_mat[i,j] > self.corr_threshold:
                        loser  = i if solo[i]<solo[j] else j
                        winner = j if loser==i else i
                        if loser in candidates:
                            candidates.remove(loser)
                            removed[loser] = (f"corr({i},{j})="
                                f"{self.corr_mat[i,j]:.3f}")
                            changed = True
                            break
                if changed: break
        return candidates, removed

    def effective_axes(self, candidates):
        """
        ★ v0.3 新機能：有効軸の自動判定
        solo_score > sens_threshold の軸を「有効軸」とする。
        """
        return [d for d in candidates
                if self.solo_score(d) > self.sens_threshold]


# ═══════════════════════════════════════════════════════
# AutoAxisSelector v0.3
# ═══════════════════════════════════════════════════════
class AutoAxisSelectorV3:
    """
    汎用パイプライン：
        任意の feat (N × D) + phases → 最良 n_select 軸を自動選択
        → E₈ / 24-cell / その他の CDSEngine でそのまま使える
    """
    def __init__(self, n_select: int, engine: CDSEngine,
                 corr_threshold=0.85, sens_threshold=0.30):
        self.n_select       = n_select
        self.engine         = engine
        self.corr_threshold = corr_threshold
        self.sens_threshold = sens_threshold
        # 結果
        self.selected_axes  = []
        self.padded_axes    = []   # ゼロパディングしたスロット数
        self.n_effective    = 0
        self.log            = []

    def fit(self, feat, times, phases, axis_names=None, verbose=True):
        N, D = feat.shape
        names = axis_names or [f'ax{d}' for d in range(D)]

        diag = AxisDiagnosticsV3(feat, times, phases,
            self.corr_threshold, self.sens_threshold)
        self.diag = diag

        # Step A: 事前冗長除去
        candidates, removed = diag.pre_filter()

        # Step A': 有効軸の判定 ★ v0.3 新機能
        effective = diag.effective_axes(candidates)
        n_eff     = len(effective)
        n_pad     = max(0, self.n_select - n_eff)

        if verbose:
            print(f"\n{'='*60}")
            print(f"AutoAxisSelector v0.3")
            print(f"候補軸 {D}本 → 有効軸 {n_eff}本 "
                  f"(閾値>{self.sens_threshold:.2f})")
            print(f"ゼロパディング: {n_pad}スロット分")
            print(f"{'='*60}")
            print(f"\n[Step A] 事前冗長除去: "
                  f"{len(removed)}軸除外 (閾値={self.corr_threshold})")
            for d,reason in removed.items():
                print(f"  axis{d:2d} {names[d]:>16}: {reason}")
            print(f"\n[Step A'] 有効軸判定 "
                  f"(sens>{self.sens_threshold:.2f}):")
            for d in candidates:
                ss = diag.solo_score(d)
                mark = '✅' if d in effective else '▷ '
                print(f"  {mark} axis{d:2d} {names[d]:>16}: "
                      f"score={ss:.4f}")

        # Step B: 有効軸の中で貪欲法
        selected  = []
        remaining = list(effective)
        self.log  = []

        if verbose:
            print(f"\n[Step B] 貪欲法 (有効軸{n_eff}本から最大"
                  f"{min(n_eff,self.n_select)}本選択)")
            print(f"  {'Step':>4} {'軸名':>16} "
                  f"{'感受性':>8} {'スコア':>8} {'S/N':>10}")
            print("  " + "-"*55)

        for step_i in range(min(n_eff, self.n_select)):
            scores = {d: diag.score(d, selected) for d in remaining}
            best_d = max(scores, key=scores.get)
            selected.append(best_d)
            remaining.remove(best_d)

            # S/N 計算（ゼロパディング込みで評価）
            sn = self._quick_sn(feat, phases, selected, n_pad)
            nm = names[best_d]
            self.log.append({'step':step_i+1,'axis':best_d,
                             'name':nm,'score':scores[best_d],'sn':sn})
            if verbose:
                print(f"  {step_i+1:>4} {nm:>16} "
                      f"{diag.sensitivity[best_d]:>8.3f} "
                      f"{scores[best_d]:>8.4f} {sn:>10.1f}×")

        if n_pad > 0 and verbose:
            print(f"\n  ★ 有効軸が {n_eff}本で n_select={self.n_select}に"
                  f"満たないため、{n_pad}スロットをゼロパディング")

        self.selected_axes = selected
        self.n_effective   = n_eff
        self.padded_axes   = list(range(n_pad))
        self.axis_names    = names

        if verbose:
            sel_names = [names[d] for d in selected]
            print(f"\n→ 選択軸 ({len(selected)}本): {sel_names}")
            print(f"→ ゼロパディング: {n_pad}スロット")
            sn_final = self._quick_sn(feat, phases, selected, n_pad)
            print(f"→ 最終 S/N: {sn_final:.1f}×")
        return selected

    def _quick_sn(self, feat, phases, selected, n_pad):
        """選択済み軸でのS/N簡易計算"""
        D_e = self.engine.D
        sub = feat[:, selected] if selected else np.zeros((len(feat),0))
        if n_pad > 0:
            sub = np.column_stack([sub, np.zeros((len(feat), n_pad))])
        sub = sub[:, :D_e] if sub.shape[1]>=D_e else np.column_stack(
            [sub, np.zeros((len(feat), D_e-sub.shape[1]))])
        _, bnd   = self.engine.normalize(sub)
        ph3      = sub[phases==3]
        ph3n, _  = self.engine.normalize(ph3, bnd)
        vids     = self.engine.assign(ph3n)
        sv       = self.engine.shape_vector(vids)
        return float(sv[np.argmax(sv)]) / (1.0/self.engine.K)

    def build_input(self, feat):
        """選択軸を抽出してゼロパディングした最終入力行列を作る"""
        D_e = self.engine.D
        sub = feat[:, self.selected_axes] if self.selected_axes \
              else np.zeros((len(feat),0))
        n_pad = max(0, self.n_select - len(self.selected_axes))
        if n_pad > 0:
            sub = np.column_stack([sub, np.zeros((len(feat), n_pad))])
        if sub.shape[1] < D_e:
            sub = np.column_stack(
                [sub, np.zeros((len(feat), D_e-sub.shape[1]))])
        return sub[:, :D_e]

    def scan(self, feat, times, window=10.0, step=2.0):
        sub = self.build_input(feat)
        _, bnd = self.engine.normalize(sub)
        return self.engine.scan(sub, times, window, step, bnd)


# ═══════════════════════════════════════════════════════
# 動作確認：v0.2 vs v0.3 比較
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    from mat_cds_sim import generate_ae_data

    df     = generate_ae_data(seed=42)
    times  = df['time'].values
    phases = df['phase'].values
    rng    = np.random.default_rng(77)
    N      = len(df)
    f4     = df[['x','y','z','log_energy']].values

    # 15候補軸（有意義5本 + 冗長2本 + ノイズ8本）
    d = {}
    d['freq_ctr']   = np.where(phases==1,rng.normal(150,30,N),
                      np.where(phases==2,rng.normal(200,25,N),
                                         rng.normal(350,20,N)))
    d['dur_log']    = np.where(phases==1,rng.normal(1.0,0.3,N),
                      np.where(phases==2,rng.normal(1.5,0.3,N),
                                         rng.normal(2.5,0.2,N)))
    d['rise_ratio'] = np.where(phases==1,rng.normal(0.1,0.05,N),
                      np.where(phases==2,rng.normal(0.3,0.1, N),
                                         rng.normal(0.7,0.1, N)))
    d['b_value']    = np.where(phases==1,rng.normal(1.5,0.2,N),
                      np.where(phases==2,rng.normal(1.2,0.2,N),
                                         rng.normal(0.7,0.1,N)))
    d['log_E']      = f4[:,3]
    d['logE_copy']  = d['log_E'] + rng.normal(0,0.05,N)
    d['x_copy']     = f4[:,0]    + rng.normal(0,0.05,N)
    for k in range(5): d[f'rand_{k}'] = rng.normal(0,1,N)
    sc = np.where(phases==1,1.,np.where(phases==2,5.,20.))
    d['nonst_1']   = rng.normal(0,1,N)*sc
    d['nonst_2']   = rng.normal(0,1,N)*sc

    names_pool = list(d.keys())
    feat_pool  = np.column_stack(list(d.values()))

    eng8 = CDSEngine(E8_UNIT, name='CDS-8D')

    print("\n" + "="*60)
    print("v0.2 vs v0.3 比較")
    print("="*60)

    # v0.3 実行
    sel_v3 = AutoAxisSelectorV3(n_select=8, engine=eng8,
                                 sens_threshold=0.30)
    sel_v3.fit(feat_pool, times, phases,
               axis_names=names_pool, verbose=True)

    # v0.2風（閾値なし：ゼロパディングなし）
    sel_v2 = AutoAxisSelectorV3(n_select=8, engine=eng8,
                                 sens_threshold=-1.0)  # 閾値=無効
    sel_v2.fit(feat_pool, times, phases,
               axis_names=names_pool, verbose=False)

    # 比較用：最初の8軸
    sn_first = sel_v3._quick_sn(feat_pool, phases, list(range(8)), 0)
    sn_v2_r  = sel_v3._quick_sn(feat_pool, phases,
                                 sel_v2.selected_axes, 0)
    sn_v3_r  = sel_v3._quick_sn(feat_pool, phases,
                                 sel_v3.selected_axes,
                                 max(0,8-sel_v3.n_effective))

    print(f"\n{'─'*50}")
    print(f"最初の8軸（無選択） : S/N = {sn_first:.1f}×")
    print(f"v0.2（閾値なし）    : S/N = {sn_v2_r:.1f}×")
    print(f"v0.3（ゼロパディング）: S/N = {sn_v3_r:.1f}×")
    print(f"v0.3 有効軸数       : {sel_v3.n_effective}/8")
    print(f"{'─'*50}")
    print(f"\n→ ゼロパディングにより S/N が改善される場合、")
    print(f"  「無意味な軸を追加するより0の方がまし」が証明される。")
