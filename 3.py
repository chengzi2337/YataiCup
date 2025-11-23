import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

# ============================================================
# 0. 物理常数与全局设置
# ============================================================
h = 6.62607015e-34
c = 2.99792458e8
k_B = 1.380649e-23

# 环境条件
T_amb = 273.15 + 15.0   # K, 环境 15 °C
h_eff = 6.0             # W/m^2/K，对流/传导换热系数（可按需要调整）

# 大气窗口简化模型参数（只看 8–13 µm）
LAM_WIN_MIN = 8.0       # µm
LAM_WIN_MAX = 13.0      # µm
N_LAM_WIN   = 450       # 大气窗口波长点数
lam_win_um  = np.linspace(LAM_WIN_MIN, LAM_WIN_MAX, N_LAM_WIN)
lam_win_m   = lam_win_um * 1e-6

# 大气在窗口里的发射率（灰体简化）
EPS_ATM = 0.3           # 晴空典型 0.2–0.4，这里取 0.3

# 太阳端总辐照度（0.3–2.5 µm 积分，经验值）
G_sun = 900.0           # W/m^2，可改成 1000 看极限情况

# PDMS 在可见光段的平均反射率（近似）
R_solar_pdms = 0.04     # PDMS 基本透明，只有少量 Fresnel 反射

# matplotlib 中文
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 在下面两个字符串中粘贴 TiO2 / PDMS 的三列表 nk 数据
#    每一行:  lambda(um)   n   k
# ============================================================

TIO2_NK_TEXT = "TiO2_0.3_25.txt"
PDMS_NK_TEXT = "pdms_nk_2.5_25.txt"


# ============================================================
# 2. 把文本 nk 数据解析成 numpy 数组
# ============================================================

def parse_nk_file(path, lam_min=0.3, lam_max=25.0):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                lam = float(parts[0])
                n   = float(parts[1])
                k   = float(parts[2])
            except ValueError:
                continue
            if lam_min <= lam <= lam_max:
                data.append([lam, n, k])
    if not data:
        raise ValueError(f"在文件 {path} 中没有解析到有效 nk 数据，请检查内容。")
    return np.array(data, dtype=float)


# 覆盖 0.3–25 µm，方便插值
TIO2_NK = parse_nk_file("TiO2_0.3_25.txt", 0.3, 25.0)
PDMS_NK = parse_nk_file("pdms_nk_2.5_25.txt", 0.3, 25.0)


# ============================================================
# 3. nk 插值工具
# ============================================================

def nk_interp(table, lam_query_um):
    """
    最近端 + 线性插值，返回复杂折射率 n_complex(λ) = n + i k
    table: (N,3) 数组 [λ, n, k]
    lam_query_um: 任意形状的波长数组 (µm)
    """
    lam = table[:, 0]
    n = table[:, 1]
    k = table[:, 2]
    lam_q = np.asarray(lam_query_um)

    n_q = np.interp(lam_q, lam, n, left=n[0], right=n[-1])
    k_q = np.interp(lam_q, lam, k, left=k[0], right=k[-1])
    return n_q + 1j * k_q


def n_tio2(lam_um):
    return nk_interp(TIO2_NK, lam_um)


def n_pdms(lam_um):
    return nk_interp(PDMS_NK, lam_um)


# ============================================================
# 4. 黑体谱 & 大气窗口制冷功率定义
# ============================================================

def M_bb_lambda(T, lam_m):
    """
    黑体谱辐射出射度 M_λ(T) [W/m^3] （兰伯特体）
    lam_m: 波长 (m)
    """
    a = 2.0 * np.pi * h * c ** 2
    b = h * c / (k_B * T)
    x = b / lam_m
    x = np.clip(x, 1e-6, 700.0)  # 防止 exp 溢出
    return a / (lam_m ** 5 * (np.exp(x) - 1.0))


def P_win_from_eps(eps_lambda, T_s):
    """
    按你模型的定义：
      P_win = ∫_8^13 ε(λ) * (1 - ε_atm) * M_bb(T_s, λ) dλ
    代表在大气窗口中，真正能“漏”到太空的那部分功率（不减天空回辐，只乘透过率）。
    """
    lam_m = lam_win_m
    I_bb = M_bb_lambda(T_s, lam_m)
    integrand = eps_lambda * (1.0 - EPS_ATM) * I_bb
    return np.trapezoid(integrand, lam_m)


# ============================================================
# 5. 单纯 PDMS 膜在大气窗口的发射率（多次反射模型）
# ============================================================

def emissivity_pdms_window(d_pdms_um):
    """
    计算“悬空 PDMS 膜”（空气 | PDMS(d) | 空气）的窗口发射率 ε(λ,d)
    使用非相干多次反射模型：
      - 顶/底界面反射率 R0
      - 膜内吸收系数 alpha = 4πk / λ
      - T, R 用教科书公式
      - ε = 1 - R - T
    d_pdms_um: PDMS 厚度 (µm)
    返回: eps(λ) 数组，长度 = N_LAM_WIN
    """
    d_m = d_pdms_um * 1e-6
    n_complex = n_pdms(lam_win_um)
    k_imag = np.imag(n_complex)

    # 界面反射（空气 / PDMS）
    r01 = (n_complex - 1.0) / (n_complex + 1.0)
    R0 = np.abs(r01) ** 2  # 单界面反射率

    # 吸收系数 alpha
    alpha = 4.0 * np.pi * k_imag / lam_win_m  # [1/m]

    # 多次反射（非相干）公式
    exp_ad  = np.exp(-alpha * d_m)
    exp_2ad = np.exp(-2.0 * alpha * d_m)

    # 为避免分母接近 0，加一个很小的正数
    den = 1.0 - R0 ** 2 * exp_2ad + 1e-30

    T = ((1.0 - R0) ** 2) * exp_ad / den
    R = R0 * (1.0 - exp_2ad) / den

    eps = 1.0 - R - T
    return np.clip(eps.real, 0.0, 1.0)


def P_win_pdms(d_pdms_um, T_s=T_amb):
    """
    纯 PDMS 膜（悬空）的窗口制冷功率
    """
    eps_pdms = emissivity_pdms_window(d_pdms_um)
    return P_win_from_eps(eps_pdms, T_s)


# ============================================================
# 6. 太阳端：TiO2 多层散射等效模型
# ============================================================

def R_solar_tio2_multi(N_tio2, d_tio2_um,
                       R_max=0.97, k_sca=0.8, alpha=1.3):
    """
    多层 TiO2 纳米散射层的平均太阳反射率模型（0.3–2.5 µm）：

      R_solar(N, d) = R_pdms +
          (R_max - R_pdms) * [ 1 - exp(-k_sca * N^alpha * d_tio2) ]

    - R_pdms  : 纯 PDMS 的太阳平均反射率（~0.04）
    - R_max   : 理论饱和反射率，白色 PDRC 涂层一般 0.94–0.98，这里取 0.97
    - k_sca   : 有效散射系数 (1/µm)，控制“爬升”速度
    - alpha   : 用来体现“多层薄膜”比“单层厚膜”更有效的增强因子，
                alpha > 1 时，在相同总厚度下，层数越多反射率越高。

    这里只是一个“拟合/经验模型”，体现趋势，而不是精确微结构模拟。
    """
    if N_tio2 <= 0 or d_tio2_um <= 0:
        return R_solar_pdms

    eff_thickness = (N_tio2 ** alpha) * d_tio2_um
    R = R_solar_pdms + (R_max - R_solar_pdms) * (1.0 - np.exp(-k_sca * eff_thickness))
    return float(np.clip(R, R_solar_pdms, R_max))


def P_solar_abs_bare():
    """
    无任何膜结构时，物体吸收的太阳功率。
    如果把裸物体视为近似黑体，则吸收 ≈ G_sun。
    """
    return G_sun


def P_solar_abs_pdms():
    """
    只加一层 PDMS 膜时，物体吸收的太阳功率（PDMS 几乎透明）
    """
    return (1.0 - R_solar_pdms) * G_sun


def P_solar_abs_stack(N_tio2, d_tio2_um):
    """
    加上 TiO2 多层结构后，物体吸收的太阳功率。
    """
    R_solar = R_solar_tio2_multi(N_tio2, d_tio2_um)
    return (1.0 - R_solar) * G_sun


# ============================================================
# 7. IR 端：TiO2 对大气窗口的“有限惩罚”
# ============================================================

def ir_penalty_factor(N_tio2, d_tio2_um,
                      eta_max=0.2, k_ir=0.5, beta_ir=1.0):
    """
    TiO2 多层对大气窗口 IR 制冷能力的“惩罚因子”：

      P_IR_stack = P_IR_PDMS * (1 - eta)

    - eta_max: 最大相对降低比例（例如 0.2 = 最多降低 20% 的 IR 制冷功率）
    - k_ir   : 控制 eta 随等效厚度增长速度
    - beta_ir: 用来体现“多层薄”比“单层厚”在 IR 上的影响更强/更弱

    经验模型：
      eta(N,d) = eta_max * [1 - exp(-k_ir * N^beta_ir * d)]
    """
    if N_tio2 <= 0 or d_tio2_um <= 0:
        return 0.0
    eff_thickness = (N_tio2 ** beta_ir) * d_tio2_um
    eta = eta_max * (1.0 - np.exp(-k_ir * eff_thickness))
    eta = float(np.clip(eta, 0.0, eta_max))
    return eta


# ============================================================
# 8. 总冷却功率 & 温降
# ============================================================

def P_cooling_stack(N_tio2, d_tio2_um,
                    T_s=T_amb, d_pdms_um=200.0):
    """
    对于给定的 (N_tio2, d_tio2_um)，总冷却功率：
        P_cool = P_IR_window_stack - P_solar_abs_stack

    【IR 端】：
      - 先用纯 PDMS 膜（200 µm）算出窗口制冷功率 P_IR_PDMS；
      - 再乘上 (1 - eta_IR)，其中 eta_IR 是 TiO2 带来的小惩罚因子。

    【太阳端】：
      - 用 TiO2 多层散射模型 R_solar_tio2_multi(N,d)，
        得到 P_solar_abs_stack(N,d)。
    """
    # 1) PDMS 的大气窗口制冷功率（基准值）
    P_ir_pdms = P_win_pdms(d_pdms_um, T_s=T_s)

    # 2) TiO2 引入的 IR 惩罚
    eta_ir = ir_penalty_factor(N_tio2, d_tio2_um)
    P_ir_stack = P_ir_pdms * (1.0 - eta_ir)

    # 3) 太阳吸收功率
    P_sun_stack = P_solar_abs_stack(N_tio2, d_tio2_um)

    # 4) 净冷却功率
    P_cool = P_ir_stack - P_sun_stack
    return P_cool, P_ir_stack, P_sun_stack


def delta_T_stack(N_tio2, d_tio2_um,
                  T_ref=T_amb, d_pdms_um=200.0):
    """
    线性近似的平衡温降：
        ΔT ≈ P_cool / h_eff
    ΔT > 0 表示表面比环境温度低。
    """
    P_cool, P_ir, P_sun = P_cooling_stack(N_tio2, d_tio2_um,
                                          T_s=T_ref, d_pdms_um=d_pdms_um)
    dT = P_cool / h_eff
    return dT, P_cool, P_ir, P_sun


# ============================================================
# 9. 扫描 TiO2 设计，寻找 ΔT 最大的 N 与 d，并画图
# ============================================================

def scan_tio2_design(T_ref=T_amb, d_pdms_um=200.0):
    # 厚度扫描范围（可以按需要调整）
    d_list = np.linspace(0.02, 2.0, 80)   # 单层 0.02–2 µm
    N_list = [1, 2, 3, 4, 5, 6]

    best = {
        "dT": -1e9,
        "N": None,
        "d": None,
        "P_cool": None,
        "P_ir": None,
        "P_sun": None,
    }

    results = {N: {"d": [], "dT": [], "P_ir": [], "P_sun": [], "P_cool": []}
               for N in N_list}

    for N in N_list:
        for d in d_list:
            dT, P_cool, P_ir, P_sun = delta_T_stack(N, d, T_ref, d_pdms_um=d_pdms_um)

            results[N]["d"].append(d)
            results[N]["dT"].append(dT)
            results[N]["P_ir"].append(P_ir)
            results[N]["P_sun"].append(P_sun)
            results[N]["P_cool"].append(P_cool)

            if dT > best["dT"]:
                best["dT"] = dT
                best["N"] = N
                best["d"] = d
                best["P_cool"] = P_cool
                best["P_ir"] = P_ir
                best["P_sun"] = P_sun

    # 转成 numpy 数组方便画图
    for N in N_list:
        for key in ["d", "dT", "P_ir", "P_sun", "P_cool"]:
            results[N][key] = np.array(results[N][key], dtype=float)

    return results, best


def plot_results(results, T_ref=T_amb):
    """
    画三张图：
      (1) P_IR_window vs d
      (2) P_solar_abs vs d
      (3) ΔT vs d
    """
    N_list = sorted(results.keys())

    # 图 1：大气窗口制冷功率 P_ir
    plt.figure(figsize=(7.5, 5))
    for N in N_list:
        d = results[N]["d"]
        P_ir = results[N]["P_ir"]
        plt.plot(d, P_ir, label=f"N = {N}")
    plt.xlabel("TiO$_2$ single-layer thickness d (µm)")
    plt.ylabel("Radiative cooling power in window $P_{IR}$ (W·m$^{-2}$)")
    plt.title(f"Radiative cooling power of TiO$_2$/PDMS structure\nin 8–13 µm atmospheric window (T = {T_ref:.1f} K)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 图 2：太阳吸收功率 P_sun
    plt.figure(figsize=(7.5, 5))
    for N in N_list:
        d = results[N]["d"]
        P_sun = results[N]["P_sun"]
        plt.plot(d, P_sun, label=f"N = {N}")
    plt.xlabel("TiO$_2$ single-layer thickness d (µm)")
    plt.ylabel("Solar absorption power $P_{solar}$ (W·m$^{-2}$)")
    plt.title("Suppression of solar absorption by TiO$_2$ multilayer structure")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 图 3：温降 ΔT
    plt.figure(figsize=(7.5, 5))
    for N in N_list:
        d = results[N]["d"]
        dT = results[N]["dT"]
        plt.plot(d, dT, label=f"N = {N}")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("TiO$_2$ single-layer thickness d (µm)")
    plt.ylabel("Estimated temperature drop ΔT (K)")
    plt.title("Overall cooling performance of TiO$_2$/PDMS structure\n(8–13 µm window + solar band)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# 10. 主程序入口
# ============================================================

if __name__ == "__main__":
    # 1) 先打印“纯 PDMS 膜”的基准性能（按 P_win 定义）
    d_pdms_um = 200.0
    P_ir_pdms = P_win_pdms(d_pdms_um, T_s=T_amb)
    P_sun_bare = P_solar_abs_bare()
    P_sun_pdms = P_solar_abs_pdms()
    P_cool_pdms = P_ir_pdms - P_sun_pdms
    dT_pdms = P_cool_pdms / h_eff

    print("========== 纯 PDMS 膜（无 TiO2）基准性能 ==========")
    print(f"PDMS 厚度 d_PDMS          = {d_pdms_um:.1f} µm")
    print(f"环境温度 T_amb           = {T_amb:.2f} K")
    print(f"大气窗口制冷功率 P_IR    = {P_ir_pdms:.2f} W/m^2")
    print(f"无膜时太阳入射功率 G_sun = {P_sun_bare:.2f} W/m^2")
    print(f"仅 PDMS 时太阳吸收 P_sun = {P_sun_pdms:.2f} W/m^2")
    print(f"净冷却功率 P_cool        = {P_cool_pdms:.2f} W/m^2")
    print(f"线性估算温降 ΔT          = {dT_pdms:.2f} K")
    print()

    # 2) 扫描 TiO2 设计，寻找 ΔT 最大的 N 和 d
    T_ref = T_amb
    results, best = scan_tio2_design(T_ref=T_ref, d_pdms_um=d_pdms_um)

    print("========== 扫描 TiO2 设计，寻找 ΔT 最大 ==========")
    print(f"最佳层数 N*               = {best['N']}")
    print(f"最佳单层厚度 d*           = {best['d']:.4f} µm")
    print(f"对应大气窗口制冷功率 P_IR = {best['P_ir']:.2f} W/m^2")
    print(f"对应太阳吸收功率 P_sun    = {best['P_sun']:.2f} W/m^2")
    print(f"对应净冷却功率 P_cool     = {best['P_cool']:.2f} W/m^2")
    print(f"对应线性温降 ΔT           = {best['dT']:.2f} K")

    # 3) 画图展示 P_IR, P_sun, ΔT 随 d 的变化
    plot_results(results, T_ref=T_ref)
