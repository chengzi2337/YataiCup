import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ===================== 1. 读入真实 n, k 数据 =====================
# 三列分别为：波长 λ(μm)，折射率 n，消光系数 k
data_str = """7.9523 1.31619 3.32E-01
8.0014 1.42903 1.74E-01
8.0511 1.38024 5.40E-02
8.1014 1.31925 2.73E-02
8.1524 1.28177 2.43E-02
8.2040 1.25491 2.83E-02
8.2562 1.23390 3.56E-02
8.3091 1.21820 4.49E-02
8.3628 1.20668 5.22E-02
8.4171 1.19571 5.81E-02
8.4721 1.18132 6.32E-02
8.5278 1.16258 6.79E-02
8.5843 1.14039 7.41E-02
8.6415 1.11398 8.41E-02
8.6995 1.07840 1.02E-01
8.7583 1.03705 1.31E-01
8.8179 0.99077 1.75E-01
8.8783 0.94243 2.40E-01
8.9395 0.90850 3.37E-01
9.0016 0.91694 4.69E-01
9.0645 0.99296 6.02E-01
9.1284 1.12582 6.88E-01
9.1931 1.27070 7.05E-01
9.2588 1.38675 6.71E-01
9.3254 1.46460 6.16E-01
9.3930 1.51165 5.62E-01
9.4615 1.53749 5.19E-01
9.5311 1.55207 4.91E-01
9.6017 1.56879 4.83E-01
9.6733 1.60177 4.91E-01
9.7461 1.66994 4.96E-01
9.8199 1.77468 4.53E-01
9.8949 1.85128 3.34E-01
9.9710 1.83876 1.94E-01
10.048 1.77344 1.06E-01
10.127 1.70999 6.35E-02
10.207 1.66025 4.21E-02
10.288 1.61895 3.12E-02
10.370 1.58348 2.52E-02
10.453 1.55340 2.08E-02
10.538 1.52658 1.83E-02
10.625 1.50134 1.81E-02
10.713 1.47663 1.82E-02
10.802 1.45099 2.14E-02
10.893 1.42981 3.13E-02
10.985 1.41565 4.01E-02
11.079 1.39727 4.13E-02
11.174 1.36630 4.66E-02
11.272 1.32958 6.75E-02
11.370 1.30103 1.12E-01
11.471 1.30308 1.67E-01
11.573 1.32585 1.93E-01
11.678 1.33199 2.02E-01
11.784 1.33215 2.22E-01
11.892 1.33290 2.27E-01
12.002 1.28572 2.44E-01
12.114 1.23134 3.43E-01
12.229 1.25325 4.89E-01
12.345 1.37130 6.40E-01
12.464 1.60462 7.21E-01
12.585 1.86355 6.09E-01
12.708 1.96263 3.61E-01
12.834 1.89414 1.74E-01
12.962 1.79653 1.02E-01
13.093 1.73460 8.96E-02
13.227 1.71230 8.48E-02
13.363 1.69554 6.27E-02
13.502 1.66481 4.22E-02
13.645 1.63533 3.49E-02
13.790 1.61279 3.34E-02
13.938 1.59113 3.87E-02
14.089 1.57651 5.18E-02
14.244 1.57656 6.28E-02
"""

data = np.loadtxt(StringIO(data_str))
lam_all = data[:, 0]   # μm
n_all = data[:, 1]
k_all = data[:, 2]

# 只取大气窗口 8–13 μm 波段
mask = (lam_all >= 8.0) & (lam_all <= 13.0)
lam_um = lam_all[mask]
n = n_all[mask]
k = k_all[mask]
lam_m = lam_um * 1e-6     # 转成 m，用于普朗克公式

# ===================== 2. 普朗克函数（黑体谱加权） =====================
h = 6.62607015e-34  # J·s
c = 2.99792458e8    # m/s
k_B = 1.380649e-23  # J/K
T = 300.0           # 取 300 K 作为辐射体温度

def planck_lambda(lam_m, T):
    """普朗克定律，返回谱辐射出射度 B_λ(λ,T)，单位 W·sr^-1·m^-3"""
    lam = lam_m
    term1 = 2 * h * c**2 / lam**5
    term2 = np.exp(h * c / (lam * k_B * T)) - 1.0
    return term1 / term2

B_lambda = planck_lambda(lam_m, T)   # 和 λ 一一对应的权重

# ===================== 3. 真实 n,k 下的光谱发射率 ε(λ,d) =====================
def emissivity_spectrum(lam_um, n_real, k_imag, d_um):
    """
    空气–PDMS–空气 结构，非相干多重反射模型。
    输入：
        lam_um : 波长数组 (μm)
        n_real, k_imag : 对应波长下的折射率、消光系数
        d_um   : 薄膜厚度 (μm)
    输出：
        eps(λ) : 该厚度下的光谱发射率数组
    """
    lam_m = lam_um * 1e-6
    d_m = d_um * 1e-6
    n_complex = n_real + 1j * k_imag

    # 入射界面反射率 (空气→PDMS)
    R0 = np.abs((n_complex - 1) / (n_complex + 1))**2

    # 吸收系数 α
    alpha = 4 * np.pi * k_imag / lam_m

    # 非相干多重反射透射率
    num = (1 - R0)**2 * np.exp(-alpha * d_m)
    den = 1 - (R0**2) * np.exp(-2 * alpha * d_m)
    T_spec = num / den

    # 能量守恒：ε = A = 1 - R - T，这里 R≈R0
    eps = 1 - R0 - T_spec
    return eps

# ===================== 4. 厚度扫描 & 普朗克加权平均发射率 =====================
# 厚度从 1 μm 扫到 300 μm，可以按需要调整
d_list = np.linspace(1, 300, 120)  # 120 个厚度点
eps_eff_list = []

for d in d_list:
    eps_lam = emissivity_spectrum(lam_um, n, k, d)
    # 普朗克加权平均发射率：
    #   eps_eff(d) = ∫ ε(λ,d) B_λ dλ / ∫ B_λ dλ
    num = np.trapz(eps_lam * B_lambda, lam_um)
    den = np.trapz(B_lambda, lam_um)
    eps_eff = num / den
    eps_eff_list.append(eps_eff)

eps_eff_arr = np.array(eps_eff_list)

# ===================== 5. 绘图：平均发射率 vs 厚度 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(7, 5))
plt.plot(d_list, eps_eff_arr, lw=2)
plt.xlabel("film thickness d / μm", fontsize=12)
plt.ylabel("8–13 μm Planck weighted mean emissivity $\\bar{\\varepsilon}(d)$", fontsize=12)
plt.title("PDMS Effect of Thin Film Thickness on Effective Emissivity", fontsize=14)
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
