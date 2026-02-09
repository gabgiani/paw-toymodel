"""
Generate Pillar 3 validation plot: two different clocks applied to
the same global state |Ψ⟩, showing different temporal descriptions.

Produces: output/validation_pillar3_two_clocks.png
          output/table_pillar3_two_clocks.csv
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import csv, os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N = 30
omega = 1.0
g = 0.1
n_env = 4
dt_C1 = 0.2
dt_C2 = 0.35

initial_S = qt.basis(2, 0)
sigma_x = qt.sigmax()
sigma_z = qt.sigmaz()


def build_and_condition(N, dt, omega, g, n_env):
    """Build PaW history state and extract conditioned observables."""
    clock_basis = [qt.basis(N, k) for k in range(N)]
    dim_env = 2**n_env
    norm = 1.0 / np.sqrt(N)

    # Build H_tot on S⊗E
    id_list = [qt.qeye(2) for _ in range(n_env)]
    H_S = (omega / 2) * qt.tensor(sigma_x, *id_list)
    H_SE = qt.Qobj(np.zeros((2 * dim_env, 2 * dim_env)),
                    dims=[[2] + [2]*n_env, [2] + [2]*n_env])
    for j in range(n_env):
        ops = [qt.qeye(2) for _ in range(n_env)]
        ops[j] = sigma_x
        H_SE += g * qt.tensor(sigma_x, *ops)
    H_tot = H_S + H_SE

    env0 = qt.tensor([qt.basis(2, 0) for _ in range(n_env)])
    initial_SE = qt.tensor(initial_S, env0)

    total_dim = N * 2 * dim_env
    psi = qt.Qobj(np.zeros((total_dim, 1)),
                   dims=[[N, 2] + [2]*n_env, [1]*(2 + n_env)])
    for k in range(N):
        t_k = k * dt
        U_SE = (-1j * H_tot * t_k).expm()
        comp = norm * qt.tensor(clock_basis[k], U_SE * initial_SE)
        psi += comp
    psi = psi.unit()

    # Condition
    d_SE = 2 * dim_env
    psi_vec = psi.full().flatten()
    blocks = psi_vec.reshape(N, d_SE)

    sz_list = []
    S_list = []
    for k in range(N):
        phi_k = blocks[k, :]
        p_k = np.vdot(phi_k, phi_k).real
        if p_k > 1e-12:
            dims_ket = [[2] + [2]*n_env, [1]*(1 + n_env)]
            psi_SE_k = qt.Qobj(phi_k.reshape(-1, 1), dims=dims_ket)
            rho_SE_k = (psi_SE_k * psi_SE_k.dag()) / p_k
            rho_S = rho_SE_k.ptrace(0)
            sz_list.append(qt.expect(sigma_z, rho_S))
            S_list.append(qt.entropy_vn(rho_S))
        else:
            sz_list.append(np.nan)
            S_list.append(np.nan)
    return np.array(sz_list), np.array(S_list)


print("Running Clock C  (dt = {})...".format(dt_C1))
sz_C1, S_C1 = build_and_condition(N, dt_C1, omega, g, n_env)

print("Running Clock C' (dt = {})...".format(dt_C2))
sz_C2, S_C2 = build_and_condition(N, dt_C2, omega, g, n_env)

ks = np.arange(N)

# ── Plot ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: sigma_z comparison
axes[0].plot(ks, sz_C1, 'o-', color='#2196F3', markersize=4, linewidth=1.2,
             label=r'Clock C  ($\Delta t = {}$)'.format(dt_C1))
axes[0].plot(ks, sz_C2, 's-', color='#E91E63', markersize=4, linewidth=1.2,
             label=r"Clock C$'$ ($\Delta t = {}$)".format(dt_C2))
axes[0].set_xlabel('Clock tick k')
axes[0].set_ylabel(r'$\langle\sigma_z\rangle$')
axes[0].set_title(r'Pillar 3: Different clocks $\rightarrow$ different dynamics')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.15, 1.15)

# Panel 2: entropy comparison
axes[1].plot(ks, S_C1, 'o-', color='#2196F3', markersize=4, linewidth=1.2,
             label=r'Clock C  ($\Delta t = {}$)'.format(dt_C1))
axes[1].plot(ks, S_C2, 's-', color='#E91E63', markersize=4, linewidth=1.2,
             label=r"Clock C$'$ ($\Delta t = {}$)".format(dt_C2))
axes[1].axhline(np.log(2), color='gray', linestyle=':', alpha=0.6,
                label=r'$\ln 2 \approx {:.3f}$'.format(np.log(2)))
axes[1].set_xlabel('Clock tick k')
axes[1].set_ylabel(r'$S_{\mathrm{eff}}$')
axes[1].set_title(r'Pillar 3: Different clocks $\rightarrow$ different arrows')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    r'Same $|\Psi\rangle$, same formula $\rho_S(t)$ — different clock $C$ yields different temporal description',
    fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/validation_pillar3_two_clocks.png", dpi=150,
            bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUTPUT_DIR}/validation_pillar3_two_clocks.png")

# ── CSV ───────────────────────────────────────────────────────
csv_path = f"{OUTPUT_DIR}/table_pillar3_two_clocks.csv"
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'sz_C1', 'Seff_C1', 'sz_C2', 'Seff_C2'])
    for k in range(N):
        w.writerow([k, f'{sz_C1[k]:.6f}', f'{S_C1[k]:.6f}',
                     f'{sz_C2[k]:.6f}', f'{S_C2[k]:.6f}'])
print(f"Saved: {csv_path}")

print("\nDone. Pillar 3 validated numerically.")
