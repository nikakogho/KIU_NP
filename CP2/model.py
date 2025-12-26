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
    """
    Keep all constants in one place (makes experiments reproducible).
    Values are placeholders for now; we'll set realistic numbers later.
    """
    return {
        # Flow split
        "mdot": 1.0,     # kg/s
        "phi": 0.5,      # fraction to branch A

        # Fluid properties
        "cp": 4000.0,    # J/(kg*K) placeholder (we'll revise)
        "m_p": 1.0,      # kg  (fluid mass per pipe segment)
        "m_cA": 1.0,     # kg  (fluid mass in cold plate A)
        "m_cB": 1.0,     # kg  (fluid mass in cold plate B)
        "m_cR": 1.0,     # kg  (fluid mass in radiator manifold)

        # Thermal capacitances (solids)
        "C_sA": 1e5,     # J/K
        "C_sB": 1e5,     # J/K
        "C_r":  5e5,     # J/K

        # Heat transfer UA values
        "UA_sA": 2e4,    # W/K heat transfer between solid and fluid in branch A
        "UA_sB": 2e4,    # W/K heat transfer between solid and fluid in branch B
        "UA_r":  3e4,    # W/K heat transfer between radiator solid and fluid

        # Radiation + solar
        "sigma": 5.670374419e-8,  # W/m^2/K^4
        "G_sun": 1361.6,          # W/m^2
        "T_bg":  3.0,             # K (background)
        "eps_r": 0.9,
        "alpha_r": 0.15,
        "A_r":  100.0,            # m^2

        "eps_p": 0.7,
        "alpha_p": 0.2,
        "A_p":  1.0,              # m^2 (pipe surface area)
        "s_r":  0.0,              # radiator sun exposure factor
        "s_p":  0.0,              # pipe sun exposure factor
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
    # Q_pipe(T) = alpha_p*A_p*G_sun*s_p - eps_p*sigma*A_p*(T^4 - T_bg^4)
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
    dy = np.zeros_like(y, dtype=float)

    # Unpack flow split
    mdot = p["mdot"]
    phi = p["phi"]
    mdot_A = phi * mdot
    mdot_B = (1.0 - phi) * mdot

    # For now, do nothing (stub). We'll fill equations next step.
    # dy[IDX["T_sA"]] = ...
    # dy[IDX["T_cA"]] = ...
    # ...

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

    cp = p["cp"]

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

    dy = f(0.0, y, p)

    assert np.all(np.isfinite(dy)), "Non-finite derivatives detected (NaN/Inf)."

    # Check expected directions
    assert dy[IDX["T_cA"]] > 0.0, "Expected coolant A to warm when T_sA > T_cA."
    assert dy[IDX["T_cB"]] > 0.0, "Expected coolant B to warm when T_sB > T_cB."

    print("Sanity check passed.")

if __name__ == "__main__":
    sanity_check_finite_and_signs()
