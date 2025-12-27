import numpy as np

# 1) State indexing (single source of truth)

def make_index():
    """
    Returns:
      idx: dict mapping variable name -> integer index in y
    State ordering must match README exactly.
    """
    names = []

    # Branch A
    names += ["T_sA", "T_cA"]
    names += [f"T_sup_A{j}" for j in range(1, 6)]
    names += [f"T_ret_A{j}" for j in range(1, 6)]

    # Branch B
    names += ["T_sB", "T_cB"]
    names += [f"T_sup_B{j}" for j in range(1, 6)]
    names += [f"T_ret_B{j}" for j in range(1, 6)]

    # Radiator
    names += ["T_cR", "T_r"]

    idx = {name: i for i, name in enumerate(names)}
    return idx, names


IDX, STATE_NAMES = make_index()
N_STATE = len(STATE_NAMES)


# 2) Parameters container

def default_params():
    import numpy as np

    # ---- Geometry assumptions (Option 2: 200 m per branch) ----
    rho = 1000.0      # kg/m^3 (water-ish)
    D   = 0.10        # m pipe inner diameter
    L_branch = 200.0  # m total (supply+return) per branch
    N_seg_total = 10  # 5 supply + 5 return
    dx = L_branch / N_seg_total

    A_cross = np.pi * (D/2)**2          # m^2
    V_seg   = A_cross * dx              # m^3
    m_p     = rho * V_seg               # kg per segment

    A_p_seg = np.pi * D * dx            # m^2 exposed surface per segment (for radiation/solar)

    # ---- Thermal / operating assumptions ----
    cp = 4180.0  # J/(kg*K)

    # 1 MW per branch with ~10K coolant rise => ~24 kg/s per branch
    mdot_total = 48.0

    # Radiator sizing: pick area so it can reject ~2 MW at a moderate temperature (~320–340K)
    A_r = 5000.0

    return {
        # Flow split
        "mdot": mdot_total,
        "phi": 0.5,

        # Fluid properties
        "cp": cp,
        "m_p": float(m_p),      # <- now geometry-based (~157 kg)
        "m_cA": 10.0,           # kg cold plate control volume (≈10 liters)
        "m_cB": 10.0,
        "m_cR": 200.0,          # kg manifold volume (bigger = smoother mixing)

        # Thermal capacitances (solids)
        "C_sA": 2.0e6,          # J/K (GPU module effective thermal mass)
        "C_sB": 2.0e6,
        "C_r":  2.0e6,          # J/K (panel warms noticeably on ~1000s timescale)

        # Heat transfer UA values
        "UA_sA": 2.0e5,         # W/K (strong cold-plate coupling)
        "UA_sB": 2.0e5,
        "UA_r":  2.0e5,         # W/K (manifold-to-panel coupling)

        # Radiation + solar
        "sigma": 5.670374419e-8,
        "G_sun": 1361.6,
        "T_bg":  3.0,

        "eps_r": 0.90,
        "alpha_r": 0.15,
        "A_r": A_r,

        "eps_p": 0.70,
        "alpha_p": 0.20,
        "A_p": float(A_p_seg),  # <- now geometry-based (~6.28 m^2 per segment)

        # Sun exposure factors (keep OFF for solver comparison runs)
        "s_r": 0.0,
        "s_p": 0.0,

        # (Optional: store geometry for debugging/printing)
        "D": D,
        "dx": dx,
        "L_branch": L_branch,
    }


# 3) Inputs (power profiles)

def P_A(t):
    # Placeholder: constant load on branch A
    return 1e6  # W

def P_B(t):
    # Placeholder: constant load on branch B
    return 1e6  # W


# 4) Helper: pipe environment heat term (Stefan-Boltzmann law + solar absorption)
def Q_pipe(T, p):
    T = float(T)
    if not np.isfinite(T):
        return np.nan
    # clamp to prevent T**4 overflow during divergence (doesn't affect normal ranges)
    T = max(1.0, min(T, 2000.0))
    return (
        p["alpha_p"] * p["A_p"] * p["G_sun"] * p["s_p"]
        - p["eps_p"] * p["sigma"] * p["A_p"] * (T**4 - p["T_bg"]**4)
    )

# 5) The ODE function: dy/dt = f(t, y)
def f(t, y, p):
    """
    Returns dy/dt as a numpy array of shape (N_STATE,).
    This will be filled in step-by-step.
    """
    if p["cp"] <= 0 or p["m_p"] <= 0 or p["m_cA"] <= 0 or p["m_cB"] <= 0 or p["m_cR"] <= 0:
        raise ValueError("Masses and cp must be positive.")
    if p["C_sA"] <= 0 or p["C_sB"] <= 0 or p["C_r"] <= 0:
        raise ValueError("Thermal capacitances must be positive.")

    dy = np.zeros_like(y, dtype=float)

    # Unpack flow split
    mdot = p["mdot"]
    phi = p["phi"]
    mdot_A = phi * mdot
    mdot_B = (1.0 - phi) * mdot

    # Server Solid Temperatures
    ## Branch A
    T_sA = y[IDX["T_sA"]]
    T_cA = y[IDX["T_cA"]]
    dy[IDX["T_sA"]] = (P_A(t) - p["UA_sA"] * (T_sA - T_cA)) / p["C_sA"]

    ## Branch B
    T_sB = y[IDX["T_sB"]]
    T_cB = y[IDX["T_cB"]]
    dy[IDX["T_sB"]] = (P_B(t) - p["UA_sB"] * (T_sB - T_cB)) / p["C_sB"]

    # Cold-Plate Coolant Temperatures

    cp = p["cp"] # J/(kg*K)

    ## Branch A Coolant
    T_sup_A5 = y[IDX["T_sup_A5"]]  # inlet to cold plate A (placeholder until pipes exist)
    dy[IDX["T_cA"]] = (
        p["UA_sA"] * (T_sA - T_cA)
        + mdot_A * cp * (T_sup_A5 - T_cA)
    ) / (p["m_cA"] * cp)

    ## Branch B Coolant
    T_sup_B5 = y[IDX["T_sup_B5"]]  # inlet to cold plate B (placeholder until pipes exist)
    dy[IDX["T_cB"]] = (
        p["UA_sB"] * (T_sB - T_cB)
        + mdot_B * cp * (T_sup_B5 - T_cB)
    ) / (p["m_cB"] * cp)

    # Pipe segments (transport + environment exchange)

    m_p = p["m_p"]     # kg per segment

    def pipe_rhs(mdot_i, T_up, T_seg):
        # m_p * cp * dT/dt = mdot_i * cp * (T_up - T_seg) + Q_pipe(T_seg)
        dT_dt = (mdot_i * cp * (T_up - T_seg) + Q_pipe(T_seg, p)) / (m_p * cp)
        return dT_dt

    ## Supply pipes: radiator -> server
    # Upstream for first supply segment is radiator manifold coolant temperature.
    T_cR = y[IDX["T_cR"]]

    # Branch A supply
    T_up = T_cR
    for j in range(1, 6):
        name = f"T_sup_A{j}"
        T_seg = y[IDX[name]]
        dy[IDX[name]] = pipe_rhs(mdot_A, T_up, T_seg)
        T_up = T_seg  # next segment sees this as upstream

    # Branch B supply
    T_up = T_cR
    for j in range(1, 6):
        name = f"T_sup_B{j}"
        T_seg = y[IDX[name]]
        dy[IDX[name]] = pipe_rhs(mdot_B, T_up, T_seg)
        T_up = T_seg

    ## Return pipes: server -> radiator
    # Upstream for first return segment is cold plate coolant temperature.
    # Branch A return
    T_up = T_cA
    for j in range(1, 6):
        name = f"T_ret_A{j}"
        T_seg = y[IDX[name]]
        dy[IDX[name]] = pipe_rhs(mdot_A, T_up, T_seg)
        T_up = T_seg

    # Branch B return
    T_up = T_cB
    for j in range(1, 6):
        name = f"T_ret_B{j}"
        T_seg = y[IDX[name]]
        dy[IDX[name]] = pipe_rhs(mdot_B, T_up, T_seg)
        T_up = T_seg

    # Mixing at radiator inlet (merge of return branches)
    
    T_ret_A5 = y[IDX["T_ret_A5"]]
    T_ret_B5 = y[IDX["T_ret_B5"]]

    mdot_tot = mdot_A + mdot_B
    if mdot_tot <= 0:
        raise ValueError("Total mass flow must be positive.")

    T_mix = (mdot_A * T_ret_A5 + mdot_B * T_ret_B5) / mdot_tot

    ## Radiator coolant manifold (well-mixed)
    T_cR = y[IDX["T_cR"]]
    T_r = y[IDX["T_r"]]

    # m_cR * cp * dT_cR/dt = mdot_tot*cp*(T_mix - T_cR) + UA_r*(T_r - T_cR)
    dy[IDX["T_cR"]] = (
        mdot_tot * cp * (T_mix - T_cR)
        + p["UA_r"] * (T_r - T_cR)
    ) / (p["m_cR"] * cp)

    ## Radiator panel (solid) with radiation + optional solar
    
    T_r = y[IDX["T_r"]]
    if not np.isfinite(T_r) or T_r < 0 or T_r > 5000:
        # 5000K is already absurd for a radiator; treat as divergence
        dy[:] = np.nan
        return dy

    # C_r * dT_r/dt = UA_r*(T_cR - T_r) - eps_r*sigma*A_r*(T_r^4 - T_bg^4) + alpha_r*A_r*G_sun*s_r
    dy[IDX["T_r"]] = (
        p["UA_r"] * (T_cR - T_r)
        - p["eps_r"] * p["sigma"] * p["A_r"] * (T_r**4 - p["T_bg"]**4)
        + p["alpha_r"] * p["A_r"] * p["G_sun"] * p["s_r"]
    ) / p["C_r"]

    return dy

def sanity_check_finite_and_signs():
    """
    Quick sanity check for early stages:
    - derivatives should be finite
    - if T_s > T_c and no inlet is hotter than coolant, then:
        dT_s/dt should be smaller than P/C (cooling exists)
        dT_c/dt should be positive (coolant heating up)
    """
    p = default_params()

    y = np.zeros(N_STATE)
    # Set some plausible temps
    y[IDX["T_sA"]] = 320.0
    y[IDX["T_cA"]] = 300.0
    y[IDX["T_sup_A5"]] = 290.0  # colder inlet

    y[IDX["T_sB"]] = 320.0
    y[IDX["T_cB"]] = 300.0
    y[IDX["T_sup_B5"]] = 290.0

    # turn off pipe radiation/solar for this test (pure advection)
    p["alpha_p"] = 0.0
    p["eps_p"] = 0.0
    p["s_p"] = 0.0

    dy = f(0.0, y, p)

    assert np.all(np.isfinite(dy)), "Non-finite derivatives detected (NaN/Inf)."

    # Check expected directions
    assert dy[IDX["T_cA"]] > 0.0, "Expected coolant A to warm when T_sA > T_cA."
    assert dy[IDX["T_cB"]] > 0.0, "Expected coolant B to warm when T_sB > T_cB."


    # set a clear upstream boundary:
    y[IDX["T_cR"]] = 280.0      # supply boundary
    y[IDX["T_cA"]] = 320.0      # return boundary A
    y[IDX["T_cB"]] = 310.0      # return boundary B

    # initialize pipes away from upstream so we can see direction
    for j in range(1, 6):
        y[IDX[f"T_sup_A{j}"]] = 400.0
        y[IDX[f"T_sup_B{j}"]] = 400.0
        y[IDX[f"T_ret_A{j}"]] = 200.0
        y[IDX[f"T_ret_B{j}"]] = 200.0

    dy = f(0.0, y, p)

    # Expect supply segments (at 400) to move DOWN toward upstream 280 -> dy negative
    assert dy[IDX["T_sup_A1"]] < 0.0
    assert dy[IDX["T_sup_B1"]] < 0.0

    # Expect return segments (at 200) to move UP toward upstream ~310/320 -> dy positive
    assert dy[IDX["T_ret_A1"]] > 0.0
    assert dy[IDX["T_ret_B1"]] > 0.0

    # radiator radiation should cool when hot (if solar is off)
    p["s_r"] = 0.0  # no solar
    y = np.zeros(N_STATE)

    # Make radiator very hot vs background; coolant cooler.
    y[IDX["T_r"]] = 600.0
    y[IDX["T_cR"]] = 300.0

    # Provide reasonable values for the merge inputs so T_mix is defined
    y[IDX["T_ret_A5"]] = 310.0
    y[IDX["T_ret_B5"]] = 310.0

    dy = f(0.0, y, p)
    assert np.isfinite(dy[IDX["T_r"]])
    assert dy[IDX["T_r"]] < 0.0, "Expected hot radiator to cool via radiation when solar=0."

    print("Radiator radiation sanity passed.")
    print("Sanity check passed.")

if __name__ == "__main__":
    sanity_check_finite_and_signs()
