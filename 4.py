import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# 0. 全局设置与超参数
# ============================================================

# matplotlib 中文字体设置
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 第三问导出的结果文件名
MODEL3_RESULTS_PATH = "model3_scan_results.npz"

# 第四问中的主物理指标：
#   "dP_tot"  -> 综合功率增益 ΔP_total（包含 IR + 太阳）【推荐】
#   "dT_IR"   -> 仅 IR 线性温差 ΔT_IR（越小越好）
PRIMARY_METRIC = "dP_tot"

# ---- 工程与成本权重（这里专门调过，保证能看见“峰值”） ----
# 材料成本在惩罚中的权重
ALPHA_COST = 0.88
# 层数（工艺复杂度）在惩罚中的权重
BETA_LAYER = 0.12
# 惩罚整体乘到综合目标上的系数 λ
LAMBDA_PENALTY = 0.8


# ============================================================
# 1. 读取第三问结果
# ============================================================

def load_model3_results(path: str = MODEL3_RESULTS_PATH):
    """
    从第三问导出的 npz 文件中读取所有扫描结果和环境参数。
    """
    data = np.load(path)

    d_list = data["d_list"]           # (nD,)
    N_list = data["N_list"]           # (nN,)

    dP_IR_mat = data["dP_IR_mat"]     # (nN, nD)
    dT_mat = data["dT_mat"]
    dP_sun_mat = data["dP_sun_mat"]
    dP_tot_mat = data["dP_tot_mat"]

    T_amb = float(data["T_amb"])
    T_ref = float(data["T_ref"])
    T_sky = float(data["T_sky"])
    h_eff = float(data["h_eff"])
    G_sun = float(data["G_sun"])
    d_pdms_um = float(data["d_pdms_um"])
    R_solar_pdms = float(data["R_solar_pdms"])

    print("===== 读取第三问数据成功 =====")
    print(f"  文件: {path}")
    print(f"  TiO2 厚度网格点数 Nd = {d_list.size}")
    print(f"  周期数个数 Nn        = {N_list.size}")
    print(f"  T_amb = {T_amb:.2f} K ({T_amb - 273.15:.2f} ℃)")
    print(f"  T_ref = {T_ref:.2f} K ({T_ref - 273.15:.2f} ℃), h_eff = {h_eff:.2f} W/m²/K")
    print("================================\n")

    return {
        "d_list": d_list,
        "N_list": N_list,
        "dP_IR_mat": dP_IR_mat,
        "dT_mat": dT_mat,
        "dP_sun_mat": dP_sun_mat,
        "dP_tot_mat": dP_tot_mat,
        "T_amb": T_amb,
        "T_ref": T_ref,
        "T_sky": T_sky,
        "h_eff": h_eff,
        "G_sun": G_sun,
        "d_pdms_um": d_pdms_um,
        "R_solar_pdms": R_solar_pdms,
    }


# ============================================================
# 2. 在 (N, d) 平面上画热力图
# ============================================================

def plot_heatmap(metric_mat: np.ndarray,
                 d_list: np.ndarray,
                 N_list: np.ndarray,
                 title: str,
                 cbar_label: str,
                 annotate_best: bool = True,
                 best_by: str = "max"):
    metric_mat = np.asarray(metric_mat)
    d_list = np.asarray(d_list)
    N_list = np.asarray(N_list)

    nN, nD = metric_mat.shape
    assert nN == N_list.size and nD == d_list.size

    D_grid, N_grid = np.meshgrid(d_list, N_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(D_grid, N_grid, metric_mat, shading="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(r"TiO$_2$ single-layer thickness $d$ ($\mu$m)")
    ax.set_ylabel(r"Number of TiO$_2$ layers $N$")
    ax.set_title(title)
    ax.set_ylim(N_list.min(), N_list.max())
    ax.set_xlim(d_list.min(), d_list.max())
    ax.grid(False)

    if annotate_best:
        if best_by == "max":
            idx_flat = np.argmax(metric_mat)
        else:
            idx_flat = np.argmin(metric_mat)
        iN_best, jd_best = np.unravel_index(idx_flat, metric_mat.shape)
        d_best = d_list[jd_best]
        N_best = N_list[iN_best]
        best_val = metric_mat[iN_best, jd_best]
        ax.plot(d_best, N_best, "ko", markersize=6,
                label=fr"Best point: N={int(N_best)}, d={d_best:.3f} $\mu$m")
        ax.legend(loc="best")
        print(f"[Heatmap] {title}")
        print(f"  最优点: N = {int(N_best)}, d = {d_best:.3f} μm, 指标值 = {best_val:.3f}\n")

    fig.tight_layout()


# ============================================================
# 3. 一维剖面：固定 N 或固定 d 的性能曲线
# ============================================================

def plot_slices(d_list: np.ndarray,
                N_list: np.ndarray,
                dT_mat: np.ndarray,
                dP_tot_mat: np.ndarray,
                primary_metric: str = PRIMARY_METRIC):
    if primary_metric == "dT_IR":
        base = -dT_mat
        base_name = "ΔT_IR（越小越好）"
    else:
        base = dP_tot_mat
        base_name = "ΔP_total"

    idx_flat = np.argmax(base)
    iN_best, jd_best = np.unravel_index(idx_flat, base.shape)
    N_best = N_list[iN_best]
    d_best = d_list[jd_best]

    print(f"[Slice] 以 {base_name} 为主指标的最优点:")
    print(f"  N_best = {int(N_best)}, d_best = {d_best:.3f} μm\n")

    # 1) 固定 N = N_best，考察随 d 的变化
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(d_list, dT_mat[iN_best, :], label="ΔT_IR (linear approximation)", linewidth=2)
    ax1.set_xlabel(r"TiO$_2$ single-layer thickness $d$ ($\mu$m)")
    ax1.set_ylabel("ΔT_IR / K")
    ax1.set_title(f"Performance vs d at fixed N = {int(N_best)}")
    ax1.axvline(d_best, color="gray", linestyle="--",
                label=fr"Optimal d = {d_best:.3f} $\mu$m")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(d_list, dP_tot_mat[iN_best, :], label="ΔP_total (IR + solar)", linewidth=2)
    ax2.set_xlabel(r"TiO$_2$ single-layer thickness $d$ ($\mu$m)")
    ax2.set_ylabel("ΔP_total / W·m$^{-2}$")
    ax2.set_title(f"Total power vs d at fixed N = {int(N_best)}")
    ax2.axvline(d_best, color="gray", linestyle="--",
                label=fr"Optimal d = {d_best:.3f} $\mu$m")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    # 2) 固定 d = d_best，考察随 N 的变化
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(N_list, dT_mat[:, jd_best], marker="o",
             label="ΔT_IR (linear approximation)", linewidth=2)
    ax3.set_xlabel(r"Number of TiO$_2$ layers $N$")
    ax3.set_ylabel("ΔT_IR / K")
    ax3.set_title(f"Performance vs N at fixed d = {d_best:.3f} $\mu$m")
    ax3.axvline(N_best, color="gray", linestyle="--",
                label=f"Optimal N = {int(N_best)}")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(N_list, dP_tot_mat[:, jd_best], marker="o",
             label="ΔP_total (IR + solar)", linewidth=2)
    ax4.set_xlabel(r"Number of TiO$_2$ layers $N$")
    ax4.set_ylabel("ΔP_total / W·m$^{-2}$")
    ax4.set_title(f"Total power vs N at fixed d = {d_best:.3f} $\mu$m")
    ax4.axvline(N_best, color="gray", linestyle="--",
                label=f"Optimal N = {int(N_best)}")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()


# ============================================================
# 4. 成本模型（材料成本） + 层数复杂度
# ============================================================

def cost_model(N: int, d_um: float,
               c_tio2: float = 3.6,
               c_pdms: float = 1.0,
               d_pdms_um: float = 200.0):
    """
    简化的“相对材料成本模型”，只做比较用：

    - PDMS 单位厚度成本记为 1；
    - 参考市售单价 + 密度，TiO2 单位体积成本约为 PDMS 的 3~4 倍，这里取 3.6；
    - 对于给定结构：
          C_raw ≈ c_tio2 * (N * d) + c_pdms * d_PDMS
      其中 d, d_PDMS 以 µm 表示，只比较相对大小。
    """
    cost_tio2 = c_tio2 * N * d_um
    cost_pdms = c_pdms * d_pdms_um
    return cost_tio2 + cost_pdms


# ============================================================
# 5. 在整个网格上计算综合评分
# ============================================================

def compute_score_grid(d_list, N_list,
                       dP_IR_mat, dT_mat, dP_sun_mat, dP_tot_mat,
                       d_pdms_um):
    """
    在整个 (N, d) 网格上计算：
        - perf_norm : 主物理指标归一化（0~1，越大越好）
        - cost_norm : 材料成本归一化（0~1，越大越贵）
        - layer_norm: 层数复杂度归一化（0~1，越大越难做）
        - penalty   : 综合惩罚项
        - S_eng     : 工程评分（10*(1-penalty)）
        - J_total   : 综合评价指标（考虑性能与成本/复杂度）
    """
    d_list = np.asarray(d_list)
    N_list = np.asarray(N_list)
    nN, nD = dP_tot_mat.shape

    # ---- 1. 主物理指标矩阵 ----
    if PRIMARY_METRIC == "dT_IR":
        metric_mat = -dT_mat   # ΔT_IR 越小越好
    else:
        metric_mat = dP_tot_mat

    perf_min = float(metric_mat.min())
    perf_max = float(metric_mat.max())
    perf_norm_mat = (metric_mat - perf_min) / (perf_max - perf_min + 1e-12)

    # ---- 2. 材料成本矩阵 ----
    cost_raw_mat = np.zeros_like(dP_tot_mat, dtype=float)
    for iN, N_val in enumerate(N_list):
        for jd, d_val in enumerate(d_list):
            cost_raw_mat[iN, jd] = cost_model(int(N_val), float(d_val),
                                              d_pdms_um=d_pdms_um)

    cost_min = float(cost_raw_mat.min())
    cost_max = float(cost_raw_mat.max())
    cost_norm_mat = (cost_raw_mat - cost_min) / (cost_max - cost_min + 1e-12)

    # ---- 3. 层数复杂度矩阵 ----
    N_min = float(N_list.min())
    N_max = float(N_list.max())
    if N_max > N_min:
        layer_norm_row = (N_list - N_min) / (N_max - N_min)
        layer_norm_mat = np.tile(layer_norm_row.reshape(-1, 1), (1, d_list.size))
    else:
        layer_norm_mat = np.zeros_like(cost_norm_mat)

    # ---- 4. 综合惩罚 & 工程评分 ----
    penalty_mat = ALPHA_COST * cost_norm_mat + BETA_LAYER * layer_norm_mat
    S_eng_mat = 10.0 * (1.0 - penalty_mat)
    S_eng_mat = np.clip(S_eng_mat, 0.0, 10.0)

    # ---- 5. 综合评价 J_total ----
    J_raw_mat = perf_norm_mat - LAMBDA_PENALTY * penalty_mat
    J_min = float(J_raw_mat.min())
    J_max = float(J_raw_mat.max())
    J_total_mat = (J_raw_mat - J_min) / (J_max - J_min + 1e-12)

    return {
        "perf_norm_mat": perf_norm_mat,
        "cost_norm_mat": cost_norm_mat,
        "layer_norm_mat": layer_norm_mat,
        "penalty_mat": penalty_mat,
        "S_eng_mat": S_eng_mat,
        "J_total_mat": J_total_mat,
        "metric_mat": metric_mat,
    }


# ============================================================
# 6. 选候选方案（按 J_total 排序）
# ============================================================

def select_candidates(d_list: np.ndarray,
                      N_list: np.ndarray,
                      dP_IR_mat: np.ndarray,
                      dT_mat: np.ndarray,
                      dP_sun_mat: np.ndarray,
                      dP_tot_mat: np.ndarray,
                      d_pdms_um: float,
                      max_candidates: int = 20):
    scores = compute_score_grid(d_list, N_list,
                                dP_IR_mat, dT_mat, dP_sun_mat, dP_tot_mat,
                                d_pdms_um)
    perf_norm_mat = scores["perf_norm_mat"]
    cost_norm_mat = scores["cost_norm_mat"]
    penalty_mat   = scores["penalty_mat"]
    S_eng_mat     = scores["S_eng_mat"]
    J_total_mat   = scores["J_total_mat"]

    d_list = np.asarray(d_list)
    N_list = np.asarray(N_list)
    nN, nD = dP_tot_mat.shape

    flat = []
    for iN in range(nN):
        for jd in range(nD):
            flat.append({
                "N": int(N_list[iN]),
                "d": float(d_list[jd]),
                "dP_IR": float(dP_IR_mat[iN, jd]),
                "dT_IR": float(dT_mat[iN, jd]),
                "dP_sun": float(dP_sun_mat[iN, jd]),
                "dP_tot": float(dP_tot_mat[iN, jd]),
                "cost_norm": float(cost_norm_mat[iN, jd]),
                "penalty": float(penalty_mat[iN, jd]),
                "S_eng": float(S_eng_mat[iN, jd]),
                "J_phys": float(perf_norm_mat[iN, jd]),
                "J_total": float(J_total_mat[iN, jd]),
            })

    flat.sort(key=lambda x: -x["J_total"])
    candidates = flat[:max_candidates]

    print("===== 候选结构列表（按综合评价 J_total 排序） =====")
    header = (
        "序号  N   d(μm)   ΔP_IR(W/m²)   ΔP_sun(W/m²)   ΔP_tot(W/m²)   "
        "ΔT_IR(K)   成本(norm)   工程评分   J_phys   penalty   J_total"
    )
    print(header)
    print("-" * len(header))
    for idx, c in enumerate(candidates, start=1):
        print(
            f"{idx:>2d}   {c['N']:>1d}   {c['d']:6.3f}   "
            f"{c['dP_IR']:10.2f}   {c['dP_sun']:10.2f}   {c['dP_tot']:10.2f}   "
            f"{c['dT_IR']:8.2f}   {c['cost_norm']:10.3f}   {c['S_eng']:8.2f}   "
            f"{c['J_phys']:7.3f}   {c['penalty']:8.3f}   {c['J_total']:7.3f}"
        )
    print("=" * len(header) + "\n")

    best_c = candidates[0]
    print(">>> 综合评价 J_total 最高的推荐结构：")
    print(f"    N* = {best_c['N']}, d* = {best_c['d']:.3f} μm")
    print(f"    ΔP_tot   = {best_c['dP_tot']:.2f} W/m²")
    print(f"    ΔP_IR    = {best_c['dP_IR']:.2f} W/m²")
    print(f"    ΔP_sun   = {best_c['dP_sun']:.2f} W/m²")
    print(f"    ΔT_IR    = {best_c['dT_IR']:.2f} K")
    print(f"    成本(norm) = {best_c['cost_norm']:.3f}")
    print(f"    工程评分   = {best_c['S_eng']:.2f}")
    print(f"    J_phys    = {best_c['J_phys']:.3f}")
    print(f"    penalty   = {best_c['penalty']:.3f}")
    print(f"    J_total   = {best_c['J_total']:.3f}\n")

    return candidates, scores


# ============================================================
# 7. 性能–成本–综合评价散点图
# ============================================================

def plot_performance_cost_scatter(candidates):
    """
    x: 成本 cost_norm（越小越好）
    y: ΔP_total（性能）
    颜色: J_total（综合评价）
    大小: J_phys（纯物理性能的归一化）
    """
    if not candidates:
        print("候选方案为空，无法绘制散点图。")
        return

    costs = np.array([c["cost_norm"] for c in candidates])
    dPtot = np.array([c["dP_tot"] for c in candidates])
    J_phys = np.array([c["J_phys"] for c in candidates])
    J_total = np.array([c["J_total"] for c in candidates])

    # 点大小：随 J_phys 变化
    sizes = 40 + 160 * (J_phys - J_phys.min()) / (J_phys.max() - J_phys.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(costs, dPtot, s=sizes, c=J_total, cmap="viridis", alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("J_total (combined metric of performance, cost, and complexity)")

    for c in candidates:
        ax.annotate(
            f"N={c['N']}, d={c['d']:.2f}",
            (c["cost_norm"], c["dP_tot"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
        )

    ax.set_xlabel("Relative cost (normalized)")
    ax.set_ylabel("ΔP_total / W·m$^{-2}$")
    ax.set_title("Performance–cost–overall-score distribution of candidate designs\n"
                 "(color = J_total, larger points indicate better physical performance)")
    ax.grid(True)
    fig.tight_layout()


# ============================================================
# 8. J_total 随成本变化（所有 N 的曲线）
# ============================================================

def plot_Jtotal_vs_cost_allN(scores, d_list, N_list):
    cost_norm_mat = scores["cost_norm_mat"]
    J_total_mat   = scores["J_total_mat"]

    fig, ax = plt.subplots(figsize=(10, 4.5))   # 横向拉长一点

    for iN, N in enumerate(N_list):
        costs = cost_norm_mat[iN, :]
        Jline = J_total_mat[iN, :]
        order = np.argsort(costs)
        ax.plot(costs[order], Jline[order], marker="o", linewidth=2, label=f"N={int(N)}")

    ax.set_xlabel("Relative cost (normalized)")
    ax.set_ylabel("J_total")

    # ====== 关键改动：缩小纵坐标范围，放大曲线间距 ======
    ax.set_ylim(0.00, 1.02)
    ax.set_yticks(np.arange(0.00, 1.02, 0.1))   # y-axis tick step = 0.1
    # ===============================================

    ax.set_title("J_total vs cost for different numbers of TiO$_2$ layers\n"
                 "After rescaling the y-axis, the peak differences become clearer")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

def plot_Jtotal_zoom(scores, d_list, N_list):
    """
    放大查看 J_total 曲线密集的部分：
        横坐标:  成本 in [0.0, 0.4]
        纵坐标:  J_total in [0.8, 1.0]
    """
    cost_norm_mat = scores["cost_norm_mat"]
    J_total_mat   = scores["J_total_mat"]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for iN, N in enumerate(N_list):
        costs = cost_norm_mat[iN, :]
        Jline = J_total_mat[iN, :]
        # 先按成本排序，使曲线连续
        order = np.argsort(costs)
        costs_sorted = costs[order]
        J_sorted     = Jline[order]

        ax.plot(costs_sorted, J_sorted, marker="o", linewidth=2, label=f"N={int(N)}")

    ax.set_xlabel("Relative cost (normalized)")
    ax.set_ylabel("J_total")

    # —— 关键：只看你关心的区间 ——
    ax.set_xlim(0.05, 0.4)   # 横坐标 0 ~ 0.4
    ax.set_ylim(0.8, 1.02)   # 纵坐标 0.8 ~ 1.02

    ax.set_title("Zoomed-in view of J_total for different numbers of TiO$_2$ layers\n"
                 "(showing only cost 0–0.4 and J_total in the 0.8–1.0 region)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


# ============================================================
# 9. 对 J_total 全局最优层数画“双轴曲线”（性能 + 综合评价）
# ============================================================

def plot_dual_axis_for_best_layer(scores, dP_tot_mat, d_list, N_list):
    cost_norm_mat = scores["cost_norm_mat"]
    J_total_mat   = scores["J_total_mat"]

    # 找到 J_total 全局最大点
    iN_star, jd_star = np.unravel_index(np.argmax(J_total_mat), J_total_mat.shape)
    N_star = int(N_list[iN_star])

    costs = cost_norm_mat[iN_star, :]
    Jline = J_total_mat[iN_star, :]
    dPline = dP_tot_mat[iN_star, :]

    order = np.argsort(costs)
    costs = costs[order]
    Jline = Jline[order]
    dPline = dPline[order]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1 = ax1.plot(costs, dPline, "o-", label="ΔP_total (performance)", linewidth=2)
    ax1.set_xlabel("Relative cost (normalized)")
    ax1.set_ylabel("ΔP_total / W·m$^{-2}$")
    ax1.grid(True)

    ax2 = ax1.twinx()
    l2 = ax2.plot(costs, Jline, "s--", color="red",
                  label="J_total (performance–cost combined metric)", linewidth=2)
    ax2.set_ylabel("J_total")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"Performance and overall score vs cost for N = {N_star}\n")
    plt.tight_layout()


# ============================================================
# 10. 主流程
# ============================================================

def main():
    # 1) 读取第三问结果
    res = load_model3_results(MODEL3_RESULTS_PATH)

    d_list = res["d_list"]
    N_list = res["N_list"]
    dP_IR_mat = res["dP_IR_mat"]
    dT_mat = res["dT_mat"]
    dP_sun_mat = res["dP_sun_mat"]
    dP_tot_mat = res["dP_tot_mat"]
    d_pdms_um = res["d_pdms_um"]

    # 2) 热力图
    plot_heatmap(
        dT_mat, d_list, N_list,
        title="Effect of TiO$_2$ structural parameters on linear IR temperature difference ΔT_IR (larger value = stronger IR penalty)",
        cbar_label="ΔT_IR / K",
        annotate_best=True,
        best_by="max",
    )
    plot_heatmap(
        dT_mat, d_list, N_list,
        title="Effect of TiO$_2$ structural parameters on linear IR temperature difference ΔT_IR (smaller is better)",
        cbar_label="ΔT_IR / K",
        annotate_best=True,
        best_by="min",
    )
    plot_heatmap(
        dP_tot_mat, d_list, N_list,
        title="Effect of TiO$_2$ structural parameters on total power gain ΔP_total",
        cbar_label="ΔP_total / W·m$^{-2}$",
        annotate_best=True,
        best_by="max",
    )

    # 3) 一维剖面
    plot_slices(d_list, N_list, dT_mat, dP_tot_mat, primary_metric=PRIMARY_METRIC)

    # 4) 候选方案 + 综合评分
    candidates, scores = select_candidates(
        d_list=d_list,
        N_list=N_list,
        dP_IR_mat=dP_IR_mat,
        dT_mat=dT_mat,
        dP_sun_mat=dP_sun_mat,
        dP_tot_mat=dP_tot_mat,
        d_pdms_um=d_pdms_um,
        max_candidates=20,
    )

    # 5) 性能–成本–综合评价散点图
    plot_performance_cost_scatter(candidates)

    # 6) 所有 N 的 J_total vs 成本曲线
    plot_Jtotal_vs_cost_allN(scores, d_list, N_list)
    plot_Jtotal_zoom(scores, d_list, N_list)

    # 7) 对 J_total 全局最优层数再画一张“双轴曲线”
    plot_dual_axis_for_best_layer(scores, dP_tot_mat, d_list, N_list)

    plt.show()


if __name__ == "__main__":
    main()
