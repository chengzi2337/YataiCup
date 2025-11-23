import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy.interpolate import UnivariateSpline

# Use English fonts for plots
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 1. Load measured nk data ==================
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
lam_data = data[:, 0]
n_data = data[:, 1]
k_data = data[:, 2]

# ================== 2. Spline smoothing for n(λ) and k(λ) ==================
# s_n, s_k control smoothness: smaller = closer to data, larger = smoother
s_n = 0.01
s_k = 0.02

spl_n = UnivariateSpline(lam_data, n_data, s=s_n)
spl_k = UnivariateSpline(lam_data, k_data, s=s_k)

lam = np.linspace(lam_data.min(), lam_data.max(), 500)
n_smooth = spl_n(lam)
k_smooth = spl_k(lam)

# Figure 1: n(λ) fit
plt.figure()
plt.plot(lam_data, n_data, "k.", label="Original n(λ) data")
plt.plot(lam, n_smooth, "b-", label="Spline-smoothed n(λ)")
plt.xlabel("Wavelength λ (μm)")
plt.ylabel("Refractive index n(λ)")
plt.title("PDMS refractive index n(λ) in 8–14 μm (spline fit)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Figure 2: k(λ) fit
plt.figure()
plt.plot(lam_data, k_data, "k.", label="Original k(λ) data")
plt.plot(lam, k_smooth, "r-", label="Spline-smoothed k(λ)")
plt.xlabel("Wavelength λ (μm)")
plt.ylabel("Extinction coefficient k(λ)")
plt.title("PDMS extinction coefficient k(λ) in 8–14 μm (spline fit)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ================== 3. Emissivity ε(λ,d) from smoothed n,k ==================
def emissivity_from_nk(lambda_um, n_real, k_imag, thickness_um_list):
    lam_m = lambda_um * 1e-6
    n_complex = n_real + 1j * k_imag
    R0 = np.abs((n_complex - 1) / (n_complex + 1))**2
    alpha = 4 * np.pi * k_imag / lam_m  # absorption coefficient

    eps_dict = {}
    for d_um in thickness_um_list:
        d_m = d_um * 1e-6
        num = (1 - R0)**2 * np.exp(-alpha * d_m)
        den = 1 - (R0**2) * np.exp(-2 * alpha * d_m)
        T = num / den
        eps = 1 - R0 - T
        eps_dict[d_um] = eps
    return eps_dict

thickness_list = [10, 50, 100, 200]
eps_dict = emissivity_from_nk(lam, n_smooth, k_smooth, thickness_list)

# Figure 3: emissivity spectra
plt.figure()
colors = plt.cm.viridis(np.linspace(0, 1, len(thickness_list)))
for c, d_um in zip(colors, thickness_list):
    plt.plot(lam, eps_dict[d_um], color=c, label=f"d = {d_um} μm")

plt.xlabel("Wavelength λ (μm)")
plt.ylabel("Emissivity ε(λ, d)")
plt.title("PDMS emissivity spectra with smoothed n,k")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
