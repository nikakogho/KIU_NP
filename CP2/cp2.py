import numpy as np
from model import IDX, N_STATE, STATE_NAMES, default_params, f, sanity_check_finite_and_signs

def g_residual(t_np1, y_guess, y_n, dt, p):
    """
    Implicit Euler residual:
      g(y) = y - y_n - dt * f(t_{n+1}, y)
    Root satisfies g(y_{n+1}) = 0.
    """
    return y_guess - y_n - dt * f(t_np1, y_guess, p)

def step_implicit_euler_fixed_point(t_n, y_n, dt, p, max_iter=100, tol=1e-8, relax=1.0):
    """
    Solve y = y_n + dt*f(t_{n+1}, y) using fixed-point iterations:
      y^{k+1} = y_n + dt*f(t_{n+1}, y^k)
    Optional relaxation:
      y^{k+1} <- (1-relax)*y^k + relax*y^{k+1}
    Returns: (y_np1, iters, converged)
    """
    t_np1 = t_n + dt
    y = y_n.copy()

    for k in range(max_iter):
        y_new = y_n + dt * f(t_np1, y, p)
        if relax != 1.0:
            y_new = (1.0 - relax) * y + relax * y_new

        if not np.all(np.isfinite(y_new)):
            return y, k + 1, False

        # convergence check (infinity norm)
        err = np.max(np.abs(y_new - y))
        y = y_new

        if err < tol:
            return y, k + 1, True

    return y, max_iter, False

def step_implicit_euler_newton_gs(t_n, y_n, dt, p, max_iter=30, tol=1e-10, eps_fd=1e-6):
    """
    Newton-Gauss-Seidel for g(y)=0:
    Sequentially update each component i:
      y_i <- y_i - g_i(y) / (d g_i / d y_i)
    using finite difference for the diagonal derivative.
    Returns: (y_np1, sweeps, converged)
    """
    t_np1 = t_n + dt
    y = y_n.copy()

    for sweep in range(max_iter):
        max_update = 0.0

        # evaluate residual at current y
        g = g_residual(t_np1, y, y_n, dt, p)

        for i in range(len(y)):
            g_i = g[i]

            # finite-difference diagonal derivative: d g_i / d y_i
            y_pert = y.copy()
            h = eps_fd * (1.0 + abs(y[i]))
            y_pert[i] += h

            g_pert = g_residual(t_np1, y_pert, y_n, dt, p)
            dg = (g_pert[i] - g_i) / h

            # avoid division by ~0
            if abs(dg) < 1e-14:
                continue

            delta = -g_i / dg
            y[i] += delta
            max_update = max(max_update, abs(delta))

            # update residual vector: recompute full g after each update (stable, slower)
            g = g_residual(t_np1, y, y_n, dt, p)

        if max_update < tol:
            return y, sweep + 1, True

    return y, max_iter, False

def simulate(method, y0, t0, tf, dt, p):
    """
    method: "fp" or "ngs"
    Returns:
      t: (N,) times
      Y: (N, N_STATE) trajectory
      stats: dict with iteration counts and convergence flags
    """
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0.copy()

    iters = np.zeros(n_steps, dtype=int)
    ok = np.zeros(n_steps, dtype=bool)

    for n in range(n_steps):
        y_n = Y[n]
        t_n = t[n]

        if method == "fp":
            y_np1, k, converged = step_implicit_euler_fixed_point(t_n, y_n, dt, p, relax=0.2)
        elif method == "ngs":
            y_np1, k, converged = step_implicit_euler_newton_gs(t_n, y_n, dt, p)
        else:
            raise ValueError("method must be 'fp' or 'ngs'")

        Y[n + 1] = y_np1
        iters[n] = k
        ok[n] = converged

        # hard safety: stop if NaN/Inf appears
        if not np.all(np.isfinite(y_np1)):
            raise FloatingPointError(f"Non-finite state at step {n}, t={t_n}")

    stats = {"iters": iters, "ok": ok}
    return t, Y, stats

if __name__ == "__main__":
    sanity_check_finite_and_signs()

    p = default_params()

    # reasonable initial temps (Kelvin)
    y0 = np.zeros(N_STATE)
    for name in STATE_NAMES:
        y0[IDX[name]] = 300.0

    # no solar for now
    p["s_r"] = 0.0
    p["s_p"] = 0.0

    t_fp, Y_fp, st_fp = simulate("fp", y0, t0=0.0, tf=20.0, dt=0.1, p=p)
    t_ngs, Y_ngs, st_ngs = simulate("ngs", y0, t0=0.0, tf=20.0, dt=0.1, p=p)

    print("FP:  ok.mean() =", st_fp["ok"].mean(), " avg iters =", st_fp["iters"].mean())
    print("NGS: ok.mean() =", st_ngs["ok"].mean(), " avg sweeps =", st_ngs["iters"].mean())
    print("Final TsA FP:", Y_fp[-1, IDX["T_sA"]], "  NGS:", Y_ngs[-1, IDX["T_sA"]])
