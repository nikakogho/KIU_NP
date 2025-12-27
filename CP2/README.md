# Orbital GPU Data Center Cooling - 2-Branch ODE Model (Implicit Euler + Nonlinear Solver Comparison)

This project models a simplified orbital GPU data center cooling loop as a **system of ordinary differential equations (ODEs)** and solves the resulting **initial value problem (IVP)** using **Implicit Euler**. The core comparison is between two nonlinear solvers used inside the *same* implicit time-stepping method:

- **Fixed-Point Iteration (FP)** for the implicit step
- **Damped Newton with Dense Finite-Difference Jacobian (ND)** for the implicit step

Both solvers are applied to the same real-world-motivated thermal model, and compared on:
- convergence robustness
- iterations/work per step
- runtime
- accuracy vs a reference solution

---

## 1) Real-World Problem

In orbit (vacuum), electronics cannot reject heat by convection. A common approach is a **pumped coolant loop** that transports heat from high-power electronics to **radiators**, which reject heat by **thermal radiation** to deep space. Radiation introduces a nonlinear term proportional to \(T^4\), and the system includes multiple coupled thermal masses with different time scales, creating a stiff nonlinear ODE system where implicit methods are appropriate.

This model represents:
- Two parallel coolant branches (A and B) that later merge
- Lumped thermal masses for GPU solids, coolant control volumes, and pipe segments
- Radiator panel thermal mass and **\(T^4\)** radiation to space
- Optional solar absorption terms (disabled in solver comparison runs)

The model is simplified and lumped: the aim is numerical-programming skill and solver comparison, not detailed spacecraft thermal design.

---

## 2) State Vector

The state consists of temperatures (Kelvin), with 2 branches and a shared radiator.

### Branch A
- `T_sA` - server solid temperature  
- `T_cA` - cold-plate coolant temperature  
- `T_sup_A1 … T_sup_A5` - supply pipe segments (radiator → server)  
- `T_ret_A1 … T_ret_A5` - return pipe segments (server → radiator)

### Branch B
- `T_sB` - server solid temperature  
- `T_cB` - cold-plate coolant temperature  
- `T_sup_B1 … T_sup_B5` - supply pipe segments  
- `T_ret_B1 … T_ret_B5` - return pipe segments  

### Radiator
- `T_cR` - radiator coolant manifold temperature  
- `T_r` - radiator panel temperature  

Full ordering used in code (`model.py`):

```text
y(t) =
[
  T_sA, T_cA,
  T_sup_A1, T_sup_A2, T_sup_A3, T_sup_A4, T_sup_A5,
  T_ret_A1, T_ret_A2, T_ret_A3, T_ret_A4, T_ret_A5,
  T_sB, T_cB,
  T_sup_B1, T_sup_B2, T_sup_B3, T_sup_B4, T_sup_B5,
  T_ret_B1, T_ret_B2, T_ret_B3, T_ret_B4, T_ret_B5,
  T_cR, T_r
]
```

---

## 3) Inputs and Parameters

### Power inputs (loads)

* `P_A(t)` - branch A heat load (W), currently constant `1e6`
* `P_B(t)` - branch B heat load (W), currently constant `1e6`

### Flow split

* `mdot` - total mass flow rate (kg/s)
* `phi` - split fraction

  * `mdot_A = phi * mdot`
  * `mdot_B = (1 - phi) * mdot`

### Radiation and environment

* `G_sun` - solar irradiance (W/m²)
* `sigma` - Stefan–Boltzmann constant
* `T_bg` - background space temperature (K)
* `eps_*`, `alpha_*`, `A_*` - emissivity, absorptivity, areas

### Lumped thermal model parameters

* `UA_sA`, `UA_sB` - coupling between solid and coolant (W/K)
* `UA_r` - coupling between radiator coolant manifold and radiator panel (W/K)
* `C_sA`, `C_sB` - solid thermal capacitances (J/K)
* `C_r` - radiator panel thermal capacitance (J/K)
* `m_cA`, `m_cB` - cold-plate coolant masses (kg)
* `m_cR` - radiator manifold coolant mass (kg)
* `m_p` - pipe segment mass (kg) computed from geometry
* `c_p` - coolant specific heat (J/kg/K)

### Geometry-based pipe mass and exposed area (in `default_params()`)

* Pipe inner diameter `D = 0.10 m`
* Total branch length `L_branch = 200 m` (supply + return per branch)
* 10 segments per branch (5 supply + 5 return) ⇒ `dx = 20 m`
* Segment mass `m_p = rho * A_cross * dx`
* Segment exposed area `A_p = pi * D * dx`

Solar exposure scaling factors (OFF during solver comparison runs):

* `s_r` (radiator solar factor) = 0.0
* `s_p` (pipes solar factor) = 0.0

---

## 4) Governing Equations (ODE System)

We solve the IVP:

$$
\dot{y}(t) = f(t, y(t)), \quad y(t_0)=y_0
$$

### 4.1 Server solids

```text
C_sA * dT_sA/dt = P_A(t) - UA_sA * (T_sA - T_cA)
C_sB * dT_sB/dt = P_B(t) - UA_sB * (T_sB - T_cB)
```

### 4.2 Cold-plate coolant volumes

```text
m_cA * c_p * dT_cA/dt =
  UA_sA * (T_sA - T_cA) + mdot_A * c_p * (T_sup_A5 - T_cA)

m_cB * c_p * dT_cB/dt =
  UA_sB * (T_sB - T_cB) + mdot_B * c_p * (T_sup_B5 - T_cB)
```

### 4.3 Pipe segments (advection + space exchange)

Each pipe segment follows:

```text
m_p * c_p * dT/dt = mdot_i * c_p * (T_upstream - T) + Q_pipe(T)
```

Supply direction: radiator manifold → cold plate
Return direction: cold plate → radiator manifold

### 4.4 Branch merge (mass-flow mixing at radiator inlet)

```text
T_mix =
  (mdot_A * T_ret_A5 + mdot_B * T_ret_B5) / (mdot_A + mdot_B)
```

### 4.5 Radiator coolant manifold (well-mixed)

```text
m_cR * c_p * dT_cR/dt =
  (mdot_A + mdot_B) * c_p * (T_mix - T_cR)
  + UA_r * (T_r - T_cR)
```

### 4.6 Radiator panel (nonlinear radiation + optional solar)

```text
C_r * dT_r/dt =
  UA_r * (T_cR - T_r)
  - eps_r * sigma * A_r * (T_r^4 - T_bg^4)
  + alpha_r * A_r * G_sun * s_r
```

### 4.7 Pipe heat exchange with environment

```text
Q_pipe(T) =
  alpha_p * A_p * G_sun * s_p
  - eps_p * sigma * A_p * (T^4 - T_bg^4)
```

### Radiative exchange assumptions

* Radiative exchange is modeled for **exposed components**: radiator panel and pipe segments.
* GPU solids and cold plates are treated as internal; direct radiation from these is neglected.
* Solar absorption is modeled separately from thermal emission.
* Solar is disabled (`s_r = s_p = 0`) during solver-comparison experiments.

---

## 5) Numerical Method

### 5.1 Implicit Euler (same timestepper for both solvers)

Implicit Euler step:

```text
y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
```

Define nonlinear residual:

```text
g(y) = y - y_n - dt * f(t_{n+1}, y) = 0
```

At each timestep, we solve `g(y)=0` using either FP or ND.

---

## 6) Nonlinear Solvers Compared

### 6.1 Fixed-Point Iteration (FP)

Iterates:

```text
y^{k+1} = y_n + dt * f(t_{n+1}, y^k)
```

With relaxation (implemented):

```text
y^{k+1} <- (1-relax)*y^k + relax*(y_n + dt*f(t_{n+1}, y^k))
```

Used settings (in code defaults):

* `max_iter=200`
* `tol=1e-8`
* `relax=0.2`

### 6.2 Damped Newton with Dense FD Jacobian (ND)

This is the current method labeled `"ngs"` in `cp2.py`, but it is not Gauss–Seidel. It is:

* Newton iterations solving `J(y) * delta = -g(y)`
* Dense Jacobian estimated via finite differences
* Backtracking line search / damping for robustness
* Step norm cap to prevent runaway updates

Key defaults:

* `max_iter=12`
* `tol_g=1e-6`, `tol_update=1e-8`
* finite-difference scale `eps_fd=1e-6`
* line search damping lower bound `damping_min=1e-3`
* step cap `max_step_norm=5e3`

---

## 7) Code Structure

### `model.py`

* Single source of truth for indexing: `IDX`, `STATE_NAMES`, `N_STATE`
* `default_params()` builds geometry-based `m_p` and `A_p`
* `f(t, y, p)` implements the ODE system
* `sanity_check_finite_and_signs()` performs basic physics/numerics checks:

  * derivatives finite
  * expected sign behavior for pipes (advection direction)
  * radiator radiation cools when solar is off

### `cp2.py`

* Implicit Euler residual `g_residual`
* Implicit step via:

  * `step_implicit_euler_fixed_point` (FP)
  * `step_implicit_euler_newton_dense` (ND; method name `"ngs"` in code)
* `simulate(method, ...)` runs the IVP
* Network visualization + animation:

  * `make_layout`, `build_network_renderer`, `animate_compare`
* Comparison suite:

  * reference run + error, iterations, ok-rate, runtime vs timestep

---

## 8) Visualization

This repository includes visualizations of the thermal network and solver behavior.

### 8.1 FP vs Newton (dense Jacobian) snapshot

![Fixed Point vs Newton (dense Jacobian)](images/fixed_point_vs_ngs.png)

### 8.2 Time evolution frames (network visualization)

![Thermal network frame 4](images/image-3.png)
![Thermal network frame 2](images/image-1.png)
![Thermal network frame 3](images/image-2.png)
![Thermal network frame 1](images/image.png)

---

## 9) Experiment Design

### Reference solution

A reference trajectory is computed using **Newton (dense Jacobian)** with a smaller timestep:

* `dt_ref = 0.5 s`
* `tf = 500 s`
* solar off: `s_r = s_p = 0`

### Tested timesteps

Both solvers are run with:

* `dt ∈ {0.5, 1.0, 2.0, 4.0} s`

### Metrics

For each `(method, dt)`:

* `runtime_s` - wall time
* `ok.mean` - fraction of steps where the nonlinear solve converged
* `avg_iters` - average iterations per timestep
* `max_rms_err` - max RMS error vs reference (excluding server solids by default)
* `diverged` - True if:

  * any non-finite values occur, or
  * not enough valid points for error estimation, or
  * `ok.mean < 0.99`

---

## 10) Results

### Summary table (from the current code runs)

| Method | dt (s) | Runtime (s) | ok.mean | Avg. iters | Max RMS error (K)      | Diverged |
| ------ | ------ | ----------- | ------- | ---------- | ---------------------- | -------- |
| FP     | 0.50   | 2.571       | 1.000   | 55.09      | 2.2552398983706557e-05 | False    |
| ND     | 0.50   | 1.328       | 1.000   | 1.03       | 0.0                    | False    |
| FP     | 1.00   | 1.313       | 1.000   | 57.86      | 0.058920700653064      | False    |
| ND     | 1.00   | 0.678       | 1.000   | 1.06       | 0.05892118837970159    | False    |
| FP     | 2.00   | 0.002       | 0.000   | 51.00      | NaN                    | True     |
| ND     | 2.00   | 0.346       | 1.000   | 1.08       | 0.1681289551635472     | False    |
| FP     | 4.00   | 0.002       | 0.000   | 25.00      | NaN                    | True     |
| ND     | 4.00   | 0.192       | 1.000   | 1.18       | 0.35803734457364106    | False    |

Interpretation:

* FP converges at smaller timesteps (`dt=0.5, 1.0`) but requires ~55–58 iterations per step.
* FP fails to converge at larger timesteps (`dt ≥ 2.0`), leading to divergence.
* Newton (dense Jacobian) converges reliably for all tested timesteps and requires ~1–2 iterations per step.

---

## 11) Conclusions

* **Implicit Euler requires a nonlinear solve each timestep**; the solver choice strongly determines stability and efficiency.
* **Fixed-Point iteration** is simple but can become impractically slow (dozens of iterations per step) and **fails for larger timesteps** on this stiff nonlinear problem.
* **Damped Newton with a dense FD Jacobian** is consistently robust on this model and converges in ~1–2 iterations per step across all tested timesteps.
* For stable timesteps (`dt=0.5` and `dt=1.0`), both solvers achieve comparable trajectories, but Newton is faster and far fewer iterations.
* Increasing timestep increases error, but Newton remains stable whereas FP diverges.

---

## 12) How to Run

### Run sanity checks (recommended)

```bash
python model.py
```

### Run animation + comparison suite

```bash
python cp2.py
```

`cp2.py` will:

* simulate FP and Newton (dense Jacobian) at a chosen `dt`
* display an animation comparing the network evolution
* run the comparison suite vs a reference solution and produce plots and a printed summary

---

## Deliverables

* **Two working codes**

  * `model.py` - ODE model, parameters, indexing, sanity checks
  * `cp2.py` - implicit Euler, FP solver, Newton solver (dense FD Jacobian), visualization, comparison suite
* **Visualizations**

  * network snapshots / frames and solver comparison animation
* **Comparison table**

  * runtime, convergence, iterations, and accuracy vs reference supporting conclusions
