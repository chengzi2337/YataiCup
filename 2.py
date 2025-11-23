import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# 0. Global matplotlib settings (English only)
# ============================================================
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]   # Use DejaVu Sans for all text
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"    # Math / superscripts
matplotlib.rcParams["axes.unicode_minus"] = False         # Proper minus sign

# ============================================================
# 1. Physical constants
# ============================================================
h = 6.62607015e-34  # Planck constant (J*s)
c = 2.99792458e8    # Speed of light (m/s)
kB = 1.380649e-23   # Boltzmann constant (J/K)

# ============================================================
# 2. Read PDMS n,k data (no header; three columns: λ  n  k)
#    Column 3 is the imaginary part / extinction coefficient k(λ)
# ============================================================
data = np.loadtxt("pdms_nk_8_14.txt")

# Columns: λ(μm), n(λ), k(λ)
lam_um_raw = data[:, 0]   # wavelength (μm)
n_raw       = data[:, 1]  # real part n(λ)
k_raw       = data[:, 2]  # imaginary part / extinction coefficient k(λ)

# Only keep the 8–13 μm atmospheric window
mask = (lam_um_raw >= 8.0) & (lam_um_raw <= 13.0)
lam_um_raw = lam_um_raw[mask]
n_raw      = n_raw[mask]
k_raw      = k_raw[mask]

# Build a fine wavelength grid for integration and plotting
lam_um = np.linspace(8.0, 13.0, 500)  # in μm
lam_m  = lam_um * 1e-6                # in m (for Planck's law)

# Interpolate n(λ), k(λ) onto the regular grid
n = np.interp(lam_um, lam_um_raw, n_raw)
k = np.interp(lam_um, lam_um_raw, k_raw)

# ============================================================
# 3. Blackbody spectral radiance I_BB(λ,T)  (units: W/m^2/m)
# ============================================================
def I_BB_lambda(lam_m, T):
    """
    Spectral blackbody radiance (hemispherical, per unit wavelength)
    lam_m : wavelength (m)
    T     : temperature (K)
    return: W/m^2/m
    """
    factor = 2.0 * np.pi * h * c**2 / (lam_m**5)
    exponent = h * c / (lam_m * kB * T)
    return factor / (np.exp(exponent) - 1.0)

# ============================================================
# 4. Compute R(λ,d), T(λ,d), ε(λ,d) from n,k using
#    an incoherent multiple-reflection model for a single layer
#
#    ñ = n + i k
#    R0(λ) = |(ñ - 1)/(ñ + 1)|^2
#    α(λ)  = 4πk/λ
#    T(λ,d)= ((1 - R0)^2 e^{-α d}) / (1 - R0^2 e^{-2α d})
#    R(λ,d)= R0 * (1 - e^{-2α d})   / (1 - R0^2 e^{-2α d})
#    ε(λ,d)= 1 - R - T
# ============================================================
def epsilon_eff_from_nk(lam_m, n_array, k_array, d):
    """
    Compute effective spectral emissivity ε(λ,d) from n(λ), k(λ) and thickness d.

    lam_m   : wavelength array (m)
    n_array : real part of refractive index
    k_array : extinction coefficient / imaginary part
    d       : film thickness (m)
    return  : ε_eff(λ,d), clipped to [0,1]
    """
    # Complex refractive index
    n_complex = n_array + 1j * k_array

    # Fresnel reflection at air (n=1) / PDMS interface (normal incidence)
    r01 = (n_complex - 1.0) / (n_complex + 1.0)
    R0 = np.abs(r01)**2

    # Absorption coefficient α(λ) = 4πk/λ  [1/m]
    alpha = 4.0 * np.pi * k_array / lam_m

    # Useful exponentials
    exp_ad  = np.exp(-alpha * d)
    exp_2ad = np.exp(-2.0 * alpha * d)

    # Incoherent transmittance
    T = ((1.0 - R0)**2) * exp_ad / (1.0 - R0**2 * exp_2ad)

    # Incoherent reflectance
    R = R0 * (1.0 - exp_2ad) / (1.0 - R0**2 * exp_2ad)

    # Emissivity from energy conservation
    eps = 1.0 - R - T

    # Numerical safety: clip to [0,1]
    eps = np.clip(eps.real, 0.0, 1.0)
    return eps

# ============================================================
# 5. Environment and atmospheric model
# ============================================================
T_amb = 288.0                        # ambient temperature (K)
I_bb_amb = I_BB_lambda(lam_m, T_amb) # blackbody spectrum at T_amb (W/m^2/m)

# Atmospheric emissivity in 8–13 μm window
# Simplified: constant value for clear-sky condition
eps_atm = 0.3 * np.ones_like(lam_m)

# ============================================================
# 6. Net cooling power in the atmospheric window P_cool0(d)
#    P_cool0(d) = ∫_8^13 ε(λ,d) [1 - ε_atm(λ)] I_BB(λ, T_amb) dλ
#    This is a "window cooling potential" index at T_s = T_amb.
# ============================================================
def P_cool0(d):
    """
    Net radiative cooling power in the 8–13 μm atmospheric window
    for a surface at T_s = T_amb.

    d : PDMS film thickness (m)
    return: P_cool0(d) in W/m^2
    """
    eps_film = epsilon_eff_from_nk(lam_m, n, k, d)
    integrand = eps_film * (1.0 - eps_atm) * I_bb_amb  # W/m^2/m
    P = np.trapezoid(integrand, lam_m)
    return P

# Convective heat transfer coefficient (for rough ΔT estimate)
h_conv = 10.0  # W/m^2/K

# ============================================================
# 7. Scan thickness: compute P_cool0(d) and ΔT(d)
# ============================================================
d_um_list = np.linspace(1.0, 200.0, 50)  # thickness (μm)
d_m_list  = d_um_list * 1e-6             # thickness (m)

P_list = []        # P_cool0(d)
DeltaT_list = []   # estimated ΔT(d) ≈ P_cool0 / h_conv

for d_m in d_m_list:
    P = P_cool0(d_m)
    P_list.append(P)
    DeltaT_list.append(P / h_conv)

P_list = np.array(P_list)
DeltaT_list = np.array(DeltaT_list)

# ============================================================
# 8. Plot: P_cool0(d) and ΔT(d)
# ============================================================
# --- Figure 1: Net cooling power in the 8–13 μm window ---
plt.figure(figsize=(7, 5))
plt.plot(d_um_list, P_list, label=r"$P_{\mathrm{cool},0}$ (window only)")
plt.xlabel("PDMS film thickness d (μm)")
plt.ylabel(r"Power (W/m$^2$)")
plt.title("Net radiative cooling power in the 8–13 μm atmospheric window")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 2: Estimated temperature drop from window cooling only ---
plt.figure(figsize=(7, 5))
plt.plot(d_um_list, DeltaT_list, label=r"Estimated $\Delta T$ (window only)")
plt.xlabel("PDMS film thickness d (μm)")
plt.ylabel(r"Estimated temperature drop (K)")
plt.title("Estimated temperature drop vs PDMS thickness\n(based on window cooling only)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 9. Spectral emissivity ε(λ) for representative thicknesses
# ============================================================
plt.figure(figsize=(7, 5))
for d_um in [10, 50, 100, 200]:
    d_m = d_um * 1e-6
    eps_spec = epsilon_eff_from_nk(lam_m, n, k, d_m)
    plt.plot(lam_um, eps_spec, label=f"d = {d_um:.0f} μm")

plt.xlabel("Wavelength λ (μm)")
plt.ylabel(r"Spectral emissivity $\varepsilon(\lambda, d)$")
plt.title("Spectral emissivity of PDMS film in the 8–13 μm window")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 10. P_win(d, Ts): window emission at arbitrary surface temperature
# ============================================================
def P_win(d, Ts):
    """
    Radiative power emitted through the 8–13 μm atmospheric window
    by a PDMS film at temperature Ts.

    d  : film thickness (m)
    Ts : surface temperature (K)
    return: P_win(d,Ts) in W/m^2
    """
    eps_film = epsilon_eff_from_nk(lam_m, n, k, d)
    I_bb_Ts = I_BB_lambda(lam_m, Ts)
    integrand = eps_film * (1.0 - eps_atm) * I_bb_Ts
    return np.trapezoid(integrand, lam_m)

# ============================================================
# 11. Plot P_win(d, Ts) for different Ts
# ============================================================
plt.figure(figsize=(7, 5))

Ts_list = [288, 300, 320, 340]  # representative surface temperatures (K)

for Ts in Ts_list:
    P_Ts = []
    for d_m in d_m_list:
        P_Ts.append(P_win(d_m, Ts))
    plt.plot(d_um_list, P_Ts, label=f"T = {Ts} K")

plt.xlabel("PDMS film thickness d (μm)")
plt.ylabel("Window radiative power (W/m$^2$)")
plt.title("Radiative power through the 8–13 μm window\nfor different film temperatures")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 12. 2D colormap: P(d, Ts) distribution
# ============================================================
Ts_grid = np.linspace(280, 340, 25)   # 280–340 K
d_um_grid = np.linspace(1.0, 200.0, 60)
d_m_grid = d_um_grid * 1e-6

P_grid = np.zeros((len(Ts_grid), len(d_um_grid)))

for i, Ts in enumerate(Ts_grid):
    for j, d_m in enumerate(d_m_grid):
        P_grid[i, j] = P_win(d_m, Ts)

D, T = np.meshgrid(d_um_grid, Ts_grid)

plt.figure(figsize=(8, 5))
pcm = plt.pcolormesh(D, T, P_grid, shading="auto")
plt.xlabel("PDMS film thickness d (μm)")
plt.ylabel("Surface temperature T (K)")
plt.title("Radiative power through the 8–13 μm window\nas a function of thickness and temperature")
cbar = plt.colorbar(pcm)
cbar.set_label("Window radiative power (W/m$^2$)")
plt.tight_layout()
plt.show()
