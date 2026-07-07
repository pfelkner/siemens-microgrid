"""Task 3 — Feasible continuous-config generator (the Benders loop's Start-`x`).

Given a microgrid instance (PV, load, outages, params) **and a fixed discrete
configuration `z`**, this produces feasible assignments of the continuous variables
`x = (P^imp, P^exp, P^ch, P^dis, E, P^peak)` — the LP feasible region of Schritt 3 in
`../QC_Ansatz_07-02.md`. Used to seed the loop (Schritt 1) and doubles as a feasibility
oracle: if the region is empty it says so (that is exactly a Benders feasibility-cut
situation, see `../doc/task1_allow_enforce.md`).

How it works (the reduction from `../doc/conversation.md`, generalized)
-------------------------------------------------------------------
With the binaries fixed, every remaining constraint is linear in `x`, so the feasible
set is a polytope. Most of `x` is *derived*, not free:

  * SoC dynamics  `E_t = E_{t-1} + η·P^ch_t·Δt − (1/η)·P^dis_t·Δt`  → defines every `E_t`.
  * Power balance `P^PV + P^imp − P^exp + P^dis − P^ch = P^load`    → defines the grid flow.

The only genuine degrees of freedom are the **battery powers**: per slot, the
magnitude `u_t ≥ 0` in the one direction the config allows (charge OR discharge OR,
if the slot is idle, nothing). Everything else is affine in `u`:

  * charge slot:    P^ch_t = u_t, P^dis_t = 0, ΔSoC = +η·Δt·u_t,   bus sign s_t = −1
  * discharge slot: P^dis_t = u_t, P^ch_t = 0, ΔSoC = −(Δt/η)·u_t, bus sign s_t = +1
  * idle slot:      u_t ≡ 0

The eliminated box bounds do not vanish — they become constraints on `u`:

  * SoC band box   → a cumulative time-chain   Elo_t ≤ SoC_init + Σ_{τ≤t} c_τ u_τ ≤ Ehi_t
  * grid box (≥0)  → per-slot bounds on `u_t` (the balance tightens the throttle)
  * grid-idle online / served-outage slot → *pins* `u_t` to an exact value (equality)

We assemble `G·u_free ≤ h` over the free battery powers, find a strictly-interior
point (Chebyshev centre via `scipy.optimize.linprog`), and draw diverse feasible
samples by **hit-and-run**. Each sample is expanded back to full `x` and verified.

Inputs (see `Instance` / `SlotConfig` / `Params`)
-------------------------------------------------
* `pv`, `load`         : per-slot PV and load (kW), length T.
* `grid_available`     : 1 = online, 0 = outage, length T.
* `config`             : list[SlotConfig], length T — the fixed discrete `z`:
      batt ∈ {"charge","discharge","idle"}, grid ∈ {"import","export","idle"},
      band ∈ {"low","mid","high"}, served (bool, only used on outage slots).
* `Params`             : Δt, η, E^max, band thresholds, P^B_nom, P^G_max, fracs,
                         SoC_init, peak_floor. Defaults match
                         `classical/deterministic_solver.py` (η = √0.9). The demo
                         overrides η = 0.9 to match conversation.md.

Output: list of dicts, each with numpy arrays p_imp, p_exp, p_ch, p_dis, soc and a
scalar p_peak — a full, verified-feasible continuous configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

EPS = 1e-7


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    dt: float = 0.25                    # h per slot
    eta: float = math.sqrt(0.9)         # per-direction efficiency (deterministic_solver default)
    e_max: float = 1000.0               # kWh
    soc_low_th: float = 100.0           # kWh, low/mid band edge
    soc_high_th: float = 900.0          # kWh, mid/high band edge
    p_bess_nom: float = 250.0           # kW nominal battery power
    p_grid_max: float = 1000.0          # kW grid cap
    frac_edge: float = 0.5              # throttle fraction in low/high band
    frac_mid: float = 1.0               # throttle fraction in mid band
    soc_init: float = 500.0             # kWh starting SoC (E_{-1})
    peak_floor: float = 0.0             # kW lower bound on P^peak
    # Cost parameters (used by the subproblem solver, not by the sampler):
    demand_charge: float = 15.0         # $/kW on the billing peak
    export_rate: float = 0.05           # $/kWh paid for grid export
    peak_mode: str = "demand_charge"    # "demand_charge" | "commit_penalty"
    penalty_rate: float = 0.0           # $/kW exceedance, only for commit_penalty

    def band_box(self, band: str) -> tuple[float, float]:
        return {
            "low": (0.0, self.soc_low_th),
            "mid": (self.soc_low_th, self.soc_high_th),
            "high": (self.soc_high_th, self.e_max),
        }[band]

    def throttle(self, band: str) -> float:
        frac = self.frac_mid if band == "mid" else self.frac_edge
        return self.p_bess_nom * frac


@dataclass
class SlotConfig:
    batt: str = "idle"        # "charge" | "discharge" | "idle"
    grid: str = "idle"        # "import" | "export" | "idle"
    band: str = "mid"         # "low" | "mid" | "high"
    served: bool = False      # only meaningful on an outage slot


@dataclass
class Instance:
    pv: np.ndarray
    load: np.ndarray
    grid_available: np.ndarray          # 1 online, 0 outage
    config: list[SlotConfig]
    params: Params = field(default_factory=Params)
    tou: np.ndarray | None = None       # $/kWh time-of-use price per slot (subproblem cost)

    def __post_init__(self):
        self.pv = np.asarray(self.pv, dtype=float)
        self.load = np.asarray(self.load, dtype=float)
        self.grid_available = np.asarray(self.grid_available, dtype=int)
        self.T = len(self.pv)
        # tou is only needed by the subproblem objective; default to a flat off-peak
        # tariff so the sampler can run without it.
        self.tou = (np.full(self.T, 0.05) if self.tou is None
                    else np.asarray(self.tou, dtype=float))
        assert len(self.load) == self.T == len(self.grid_available) == len(self.config), \
            "pv, load, grid_available, config must all have length T"


class Infeasible(Exception):
    """Raised when the fixed config admits no feasible continuous continuation."""


# --------------------------------------------------------------------------- #
# Polytope construction (reduce to free battery powers)
# --------------------------------------------------------------------------- #
@dataclass
class _Slot:
    c: float          # SoC coefficient: ΔSoC_t = c * u_t
    s: int            # bus sign: (P^dis - P^ch) = s * u_t   (+1 dis, -1 ch, 0 idle)
    umax: float       # throttle upper bound on u_t
    fixed: float | None   # None = free, else pinned value


def _classify_slots(inst: Instance) -> list[_Slot]:
    """Per slot: SoC coeff, bus sign, throttle, and whether the balance pins u_t."""
    p = inst.params
    slots: list[_Slot] = []
    for t in range(inst.T):
        cfg = inst.config[t]
        umax = p.throttle(cfg.band)
        if cfg.batt == "charge":
            c, s = p.eta * p.dt, -1
        elif cfg.batt == "discharge":
            c, s = -p.dt / p.eta, +1
        else:  # idle
            c, s, umax = 0.0, 0, 0.0

        rhs = inst.load[t] - inst.pv[t]      # net demand at the bus (before battery)
        online = inst.grid_available[t] == 1

        # Cases that PIN u_t to an exact value (equality on the balance):
        #   online + grid idle          → P^imp = P^exp = 0 → s*u = rhs
        #   outage  + served            → battery must exactly cover load → s*u = rhs
        # (outage forces P^imp = P^exp = 0 regardless of the grid bit.)
        pinned = (online and cfg.grid == "idle") or ((not online) and cfg.served)
        fixed: float | None = None
        if cfg.batt == "idle":
            fixed = 0.0
            if pinned and abs(rhs) > EPS:
                raise Infeasible(
                    f"slot {t}: balance needs battery {rhs:+.3f} kW but battery is idle")
        elif pinned:
            v = rhs / s
            if v < -EPS or v > umax + EPS:
                raise Infeasible(
                    f"slot {t}: balance pins battery to {v:.3f} kW, outside [0,{umax}]")
            fixed = min(max(v, 0.0), umax)
        slots.append(_Slot(c=c, s=s, umax=umax, fixed=fixed))
    return slots


def _rows(inst: Instance, slots: list[_Slot]) -> tuple[np.ndarray, np.ndarray]:
    """All inequality rows a·u ≤ b over the FULL u∈R^T (fixed slots substituted later)."""
    p, T = inst.params, inst.T
    A: list[np.ndarray] = []
    b: list[float] = []

    def row(a: np.ndarray, rhs: float):
        A.append(a)
        b.append(rhs)

    e = np.eye(T)
    for t in range(T):
        st, cfg = slots[t], inst.config[t]
        # throttle box  0 ≤ u_t ≤ umax
        row(-e[t], 0.0)
        row(e[t], st.umax)

        # SoC band chain:  Elo ≤ soc_init + Σ_{τ≤t} c_τ u_τ ≤ Ehi
        cum = np.array([slots[tau].c if tau <= t else 0.0 for tau in range(T)])
        elo, ehi = p.band_box(cfg.band)
        row(cum, ehi - p.soc_init)      # upper band edge
        row(-cum, p.soc_init - elo)     # lower band edge

        # grid box, only where a grid flow actually exists (online, not grid-idle)
        online = inst.grid_available[t] == 1
        if online and cfg.grid == "import":
            # P^imp = rhs - s*u ∈ [0, Pgmax]
            rhs = inst.load[t] - inst.pv[t]
            row(st.s * e[t], rhs)                       # P^imp ≥ 0
            row(-st.s * e[t], p.p_grid_max - rhs)       # P^imp ≤ Pgmax
        elif online and cfg.grid == "export":
            # P^exp = s*u - rhs ∈ [0, Pgmax]
            rhs = inst.load[t] - inst.pv[t]
            row(-st.s * e[t], -rhs)                      # P^exp ≥ 0
            row(st.s * e[t], p.p_grid_max + rhs)         # P^exp ≤ Pgmax
        # grid-idle online & served-outage were already pinned; outage-unserved has
        # no grid flow and a free residual → no grid row.
    return np.array(A), np.array(b)


def _reduce(A: np.ndarray, b: np.ndarray, slots: list[_Slot]):
    """Substitute fixed slots → inequalities over the free coordinates only."""
    free = [t for t, s in enumerate(slots) if s.fixed is None]
    xfix = np.array([s.fixed if s.fixed is not None else 0.0 for s in slots])
    Gf = A[:, free]
    hf = b - A @ xfix                      # move fixed contribution to RHS
    # Drop rows with no free coefficient — they are constant feasibility checks.
    keep, const_ok = [], True
    for i in range(len(hf)):
        if np.all(np.abs(Gf[i]) < EPS):
            if hf[i] < -1e-6:
                const_ok = False
        else:
            keep.append(i)
    if not const_ok:
        raise Infeasible("a fixed/idle slot violates a grid or SoC bound")
    return np.array(free), Gf[keep], hf[keep], xfix


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _chebyshev_center(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Strictly-interior point: max r s.t. G x + r||G_i|| ≤ h. Raises if empty."""
    d = G.shape[1]
    norms = np.linalg.norm(G, axis=1)
    A_ub = np.hstack([G, norms[:, None]])
    c = np.zeros(d + 1)
    c[-1] = -1.0                            # maximise r
    bounds = [(None, None)] * d + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=h, bounds=bounds, method="highs")
    if not res.success or res.x[-1] < 1e-9:
        raise Infeasible("polytope empty or lower-dimensional (no interior point)")
    return res.x[:d]


def _hit_and_run(G, h, x0, n, rng) -> list[np.ndarray]:
    """Draw n feasible points by hit-and-run random walk from interior x0."""
    d = len(x0)
    out, x = [], x0.copy()
    Gd = G
    for _ in range(n):
        dirn = rng.standard_normal(d)
        dirn /= np.linalg.norm(dirn) + EPS
        gd = Gd @ dirn                       # step limits: G(x + t d) ≤ h
        slack = h - Gd @ x
        t_hi, t_lo = np.inf, -np.inf
        with np.errstate(divide="ignore"):
            ratios = slack / gd
        t_hi = min(t_hi, ratios[gd > EPS].min(initial=np.inf))
        t_lo = max(t_lo, ratios[gd < -EPS].max(initial=-np.inf))
        if not np.isfinite(t_hi):
            t_hi = 0.0
        if not np.isfinite(t_lo):
            t_lo = 0.0
        t = rng.uniform(t_lo, t_hi)
        x = x + t * dirn
        out.append(x.copy())
    return out


# --------------------------------------------------------------------------- #
# Reconstruction + verification
# --------------------------------------------------------------------------- #
def _reconstruct(inst: Instance, slots: list[_Slot], u_full: np.ndarray) -> dict:
    p, T = inst.params, inst.T
    p_ch = np.zeros(T); p_dis = np.zeros(T)
    p_imp = np.zeros(T); p_exp = np.zeros(T)
    soc = np.zeros(T)
    e_prev = p.soc_init
    for t in range(T):
        cfg, u = inst.config[t], u_full[t]
        if cfg.batt == "charge":
            p_ch[t] = u
        elif cfg.batt == "discharge":
            p_dis[t] = u
        soc[t] = e_prev + p.eta * p_ch[t] * p.dt - p_dis[t] / p.eta * p.dt
        e_prev = soc[t]

        bus = p_dis[t] - p_ch[t]             # battery injection to bus
        rhs = inst.load[t] - inst.pv[t]
        if inst.grid_available[t] == 1:
            if cfg.grid == "import":
                p_imp[t] = rhs - bus
            elif cfg.grid == "export":
                p_exp[t] = bus - rhs
        # outage / grid-idle: both stay 0
    p_peak = max(p.peak_floor, float(p_imp.max()) if T else 0.0)
    return dict(p_imp=p_imp, p_exp=p_exp, p_ch=p_ch, p_dis=p_dis, soc=soc, p_peak=p_peak)


def verify(inst: Instance, x: dict, tol: float = 1e-5) -> None:
    """Assert a reconstructed x is feasible. The runnable self-check for this module."""
    p, T = inst.params, inst.T
    for t in range(T):
        cfg = inst.config[t]
        elo, ehi = p.band_box(cfg.band)
        thr = p.throttle(cfg.band)
        assert -tol <= x["soc"][t] <= p.e_max + tol, f"SoC out of [0,E_max] @ {t}"
        assert elo - tol <= x["soc"][t] <= ehi + tol, f"SoC out of band {cfg.band} @ {t}"
        assert -tol <= x["p_ch"][t] <= thr + tol and -tol <= x["p_dis"][t] <= thr + tol, \
            f"battery power over throttle @ {t}"
        assert -tol <= x["p_imp"][t] <= p.p_grid_max + tol, f"import out of box @ {t}"
        assert -tol <= x["p_exp"][t] <= p.p_grid_max + tol, f"export out of box @ {t}"
        assert x["p_peak"] >= x["p_imp"][t] - tol, f"peak below import @ {t}"
        if inst.grid_available[t] == 1:
            bal = (inst.pv[t] + x["p_imp"][t] - x["p_exp"][t]
                   + x["p_dis"][t] - x["p_ch"][t] - inst.load[t])
            assert abs(bal) <= tol, f"power balance violated @ {t}: {bal:.2e}"
        else:
            assert abs(x["p_imp"][t]) <= tol and abs(x["p_exp"][t]) <= tol, \
                f"grid nonzero during outage @ {t}"
            if cfg.served:
                resid = inst.pv[t] + x["p_dis"][t] - x["p_ch"][t] - inst.load[t]
                assert abs(resid) <= tol, f"served outage not balanced @ {t}: {resid:.2e}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def feasible_configs(inst: Instance, n: int = 5, seed: int = 0) -> list[dict]:
    """Return up to `n` verified-feasible continuous configs for the fixed `z`.

    Raises `Infeasible` if the fixed discrete config has no continuous continuation
    (this is precisely a Benders feasibility-cut event; see ../doc/task1_allow_enforce.md).
    """
    slots = _classify_slots(inst)                       # may raise Infeasible
    A, b = _rows(inst, slots)
    free, G, h, xfix = _reduce(A, b, slots)             # may raise Infeasible

    rng = np.random.default_rng(seed)
    if len(free) == 0:                                  # everything pinned/idle
        u_full = xfix
        x = _reconstruct(inst, slots, u_full)
        verify(inst, x)
        return [x]

    x0 = _chebyshev_center(G, h)                        # may raise Infeasible
    samples = [x0] + _hit_and_run(G, h, x0, n - 1, rng) if n > 1 else [x0]

    out = []
    for uf in samples[:n]:
        u_full = xfix.copy()
        u_full[free] = uf
        x = _reconstruct(inst, slots, u_full)
        verify(inst, x)                                 # every sample is checked
        out.append(x)
    return out


# --------------------------------------------------------------------------- #
# Demo / self-check: the T=3 instance from ../doc/conversation.md
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # conversation.md uses η = 0.9 directly (coeffs 0.225 = η·Δt, 0.2778 = Δt/η).
    params = Params(eta=0.9, soc_init=120.0)
    inst = Instance(
        pv=[100, 400, 150],
        load=[300, 200, 350],
        grid_available=[1, 1, 1],
        config=[
            SlotConfig(batt="discharge", grid="import", band="mid"),  # t0
            SlotConfig(batt="charge",    grid="export", band="mid"),  # t1
            SlotConfig(batt="discharge", grid="import", band="mid"),  # t2
        ],
        params=params,
    )
    xs = feasible_configs(inst, n=200, seed=1)
    assert len(xs) == 200, "expected 200 samples"

    # conversation.md's reduced polytope: θ_0 = P^dis_0 ≤ 72 (SoC floor binds, not 250);
    # θ_1, θ_2 ≤ 200 (balance tightens the 250 throttle to 200). Check every sample.
    for x in xs:
        assert x["p_dis"][0] <= 72.0 + 1e-4, f"θ0={x['p_dis'][0]:.3f} > 72"
        assert x["p_ch"][1] <= 200.0 + 1e-4
        assert x["p_dis"][2] <= 200.0 + 1e-4
    # the SoC chain must actually be exercised (some sample charges at t1)
    assert max(x["p_ch"][1] for x in xs) > 1.0, "hit-and-run never explored t1 charge"
    # verify() already ran on each; balance/bounds all hold.
    print(f"[demo] OK — {len(xs)} feasible samples, all within the reduced polytope.")
    print(f"       θ0 range [{min(x['p_dis'][0] for x in xs):.2f}, "
          f"{max(x['p_dis'][0] for x in xs):.2f}] (bound 72)")

    # Infeasibility detection: force battery idle during a deficit outage that must be
    # served → no continuation → Infeasible (a feasibility-cut situation).
    bad = Instance(
        pv=[0.0], load=[300.0], grid_available=[0],
        config=[SlotConfig(batt="idle", grid="idle", band="mid", served=True)],
        params=Params(eta=0.9, soc_init=500.0),
    )
    try:
        feasible_configs(bad, n=1)
        raise AssertionError("expected Infeasible for unservable idle outage")
    except Infeasible as ex:
        print(f"[demo] OK — infeasibility detected: {ex}")


if __name__ == "__main__":
    _demo()
