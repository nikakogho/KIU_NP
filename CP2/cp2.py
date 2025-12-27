import numpy as np
from model import IDX, N_STATE, P_A, P_B, STATE_NAMES, default_params, f, sanity_check_finite_and_signs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

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

def make_layout():
    """
    Returns:
      pos: dict name -> (x,y)
      edges: list of (name_u, name_v, kind) where kind is 'pipe' or 'link'
    """
    pos = {}

    # Anchor positions
    pos["T_sA"] = (-2.0,  0.6)
    pos["T_cA"] = (-1.3,  0.1)

    pos["T_sB"] = (-2.0, -0.6)
    pos["T_cB"] = (-1.3, -0.1)

    pos["T_cR"] = ( 2.6,  0.0)
    pos["T_r"]  = ( 3.4,  0.0)

    def arc_points(start, end, bulge, n=5):
        """
        Make n points between start and end with a sine bulge.
        Returns list of length n with coordinates for segment nodes 1..n.
        """
        x0, y0 = start
        x1, y1 = end
        pts = []
        for j in range(1, n+1):
            s = j/(n+1)  # exclude endpoints
            x = (1-s)*x0 + s*x1
            y = (1-s)*y0 + s*y1
            # perpendicular-ish bulge in y
            y += bulge * np.sin(np.pi*s)
            pts.append((x, y))
        return pts

    # Supply: radiator -> cold plates (A top arc, B bottom arc)
    supA = arc_points(pos["T_cR"], pos["T_cA"], bulge=+0.9, n=5)
    supB = arc_points(pos["T_cR"], pos["T_cB"], bulge=-0.9, n=5)

    for j in range(1, 6):
        pos[f"T_sup_A{j}"] = supA[j-1]
        pos[f"T_sup_B{j}"] = supB[j-1]

    # Return: cold plates -> radiator (A bottom-ish arc, B top-ish arc)
    retA = arc_points(pos["T_cA"], pos["T_cR"], bulge=-0.7, n=5)
    retB = arc_points(pos["T_cB"], pos["T_cR"], bulge=+0.7, n=5)

    for j in range(1, 6):
        pos[f"T_ret_A{j}"] = retA[j-1]
        pos[f"T_ret_B{j}"] = retB[j-1]

    edges = []

    # Branch A supply chain: T_cR -> T_sup_A1 -> ... -> T_sup_A5 -> T_cA
    edges.append(("T_cR", "T_sup_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_A{j}", f"T_sup_A{j+1}", "pipe"))
    edges.append(("T_sup_A5", "T_cA", "pipe"))

    # Branch A return chain: T_cA -> T_ret_A1 -> ... -> T_ret_A5 -> T_cR
    edges.append(("T_cA", "T_ret_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_A{j}", f"T_ret_A{j+1}", "pipe"))
    edges.append(("T_ret_A5", "T_cR", "pipe"))

    # Branch B supply chain
    edges.append(("T_cR", "T_sup_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_B{j}", f"T_sup_B{j+1}", "pipe"))
    edges.append(("T_sup_B5", "T_cB", "pipe"))

    # Branch B return chain
    edges.append(("T_cB", "T_ret_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_B{j}", f"T_ret_B{j+1}", "pipe"))
    edges.append(("T_ret_B5", "T_cR", "pipe"))

    # Couplings (non-pipe links)
    edges.append(("T_sA", "T_cA", "link"))
    edges.append(("T_sB", "T_cB", "link"))
    edges.append(("T_cR", "T_r", "link"))

    return pos, edges

def build_network_renderer(pos, edges, norm, *,
                           figsize=(14, 8),
                           label_fontsize=8,
                           label_offset=(0.05, 0.05),
                           show_labels=True,
                           show_boxes=True):
    """
    Creates a figure + artists once, and returns an update(y, title_text) function.
    Reusable for snapshot + animation.

    Returns:
      fig, ax, update_fn
    """
    node_names = list(pos.keys())
    node_idx = np.array([IDX[n] for n in node_names], dtype=int)

    xs = np.array([pos[n][0] for n in node_names], dtype=float)
    ys = np.array([pos[n][1] for n in node_names], dtype=float)

    # Precompute edge segments + endpoint index pairs
    pipe_segs, pipe_pairs = [], []
    link_segs, link_pairs = [], []

    for u, v, kind in edges:
        seg = [(pos[u][0], pos[u][1]), (pos[v][0], pos[v][1])]
        if kind == "pipe":
            pipe_segs.append(seg)
            pipe_pairs.append((IDX[u], IDX[v]))
        else:
            link_segs.append(seg)
            link_pairs.append((IDX[u], IDX[v]))

    pipe_pairs = np.array(pipe_pairs, dtype=int) if pipe_pairs else np.zeros((0, 2), dtype=int)
    link_pairs = np.array(link_pairs, dtype=int) if link_pairs else np.zeros((0, 2), dtype=int)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Optional boxes for GPU/radiator
    if show_boxes:
        ax.add_patch(Rectangle((-2.4, -1.05), 1.5, 2.1, fill=False, linewidth=2))
        ax.text(-2.35, 1.05, "GPU MODULE", fontsize=12, va="bottom")

        ax.add_patch(Rectangle((2.25, -0.55), 1.45, 1.1, fill=False, linewidth=2))
        ax.text(2.3, 0.6, "RADIATOR", fontsize=12, va="bottom")

    # Lines
    lc_pipe = LineCollection(pipe_segs, linewidths=6)
    lc_pipe.set_norm(norm)
    ax.add_collection(lc_pipe)

    lc_link = LineCollection(link_segs, linewidths=2, linestyles="dashed")
    lc_link.set_norm(norm)
    ax.add_collection(lc_link)

    # Node sizes
    sizes = []
    for name in node_names:
        if name in ("T_sA", "T_cA", "T_sB", "T_cB"):
            sizes.append(220)
        elif name in ("T_cR", "T_r"):
            sizes.append(260)
        else:
            sizes.append(120)

    sc = ax.scatter(xs, ys, s=sizes, zorder=3)
    sc.set_norm(norm)

    # Labels
    if show_labels:
        dx, dy0 = label_offset
        for name in node_names:
            x, yy = pos[name]
            ax.text(
                x + dx, yy + dy0, name,
                fontsize=label_fontsize,
                ha="left", va="bottom",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )

    title = ax.set_title("Thermal state snapshot")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # breathing room
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    ax.set_xlim(xmin - 0.9, xmax + 0.9)
    ax.set_ylim(ymin - 0.9, ymax + 0.9)

    # Colorbar once
    mappable = plt.cm.ScalarMappable(norm=norm)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, label="Temperature (K)")

    def update_with_y(y, title_text=None):
        # edge colors = average endpoint temps
        if len(pipe_pairs):
            pipe_vals = 0.5 * (y[pipe_pairs[:, 0]] + y[pipe_pairs[:, 1]])
            lc_pipe.set_array(pipe_vals)
        if len(link_pairs):
            link_vals = 0.5 * (y[link_pairs[:, 0]] + y[link_pairs[:, 1]])
            lc_link.set_array(link_vals)

        # node colors
        sc.set_array(y[node_idx])

        if title_text is not None:
            title.set_text(title_text)

        return lc_pipe, lc_link, sc, title

    return fig, ax, update_with_y

def plot_snapshot(t, Y, pos, edges, i=-1, *,
                  scale_exclude_servers=True):
    # choose color scale
    if scale_exclude_servers:
        scale_names = [n for n in pos.keys() if n not in ("T_sA", "T_sB")]
        scale_idx = np.array([IDX[n] for n in scale_names], dtype=int)
        vmin = float(np.min(Y[:, scale_idx]))
        vmax = float(np.max(Y[:, scale_idx]))
    else:
        vmin = float(np.min(Y))
        vmax = float(np.max(Y))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig, ax, update = build_network_renderer(pos, edges, norm)

    if i < 0:
        i = len(t) - 1
    update(Y[i], title_text=f"t={t[i]:.2f}s")
    plt.show()

def animate_network(t, Y, pos, edges, *,
                    stride=1,
                    interval_ms=60,
                    scale_exclude_servers=True):
    # color scale
    if scale_exclude_servers:
        scale_names = [n for n in pos.keys() if n not in ("T_sA", "T_sB")]
        scale_idx = np.array([IDX[n] for n in scale_names], dtype=int)
        vmin = float(np.min(Y[:, scale_idx]))
        vmax = float(np.max(Y[:, scale_idx]))
    else:
        vmin = float(np.min(Y))
        vmax = float(np.max(Y))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig, ax, update = build_network_renderer(pos, edges, norm)

    frame_indices = np.arange(0, len(t), stride, dtype=int)

    def anim_update(k):
        i = frame_indices[k]
        return update(Y[i], title_text=f"t={t[i]:.2f}s")

    anim = FuncAnimation(
        fig, anim_update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,
        repeat=True
    )
    plt.show()
    return anim

def Q_rad_surface(T, eps, A, p):
    return eps * p["sigma"] * A * (T**4 - p["T_bg"]**4)

import time

def run_one(method, y0, p, *, t0=0.0, tf=2000.0, dt=1.0,
            fp_relax=0.2, fp_max_iter=200, ngs_max_iter=50,
            tol_fp=1e-8, tol_ngs=1e-10):
    """
    Runs one simulation and returns a dict of metrics for comparison.
    """
    t_start = time.perf_counter()

    if method == "fp":
        # temporarily wrap simulate to pass fp settings without changing simulate signature too much
        t, Y, st = simulate_fp(y0, t0, tf, dt, p,
                              relax=fp_relax, max_iter=fp_max_iter, tol=tol_fp)
    elif method == "ngs":
        t, Y, st = simulate_ngs(y0, t0, tf, dt, p,
                                max_iter=ngs_max_iter, tol=tol_ngs)
    else:
        raise ValueError("method must be 'fp' or 'ngs'")

    elapsed = time.perf_counter() - t_start

    ok = st["ok"]
    iters = st["iters"]

    metrics = {
        "method": method,
        "dt": dt,
        "tf": tf,
        "steps": len(t) - 1,
        "ok_rate": float(ok.mean()),
        "fail_steps": int((~ok).sum()),
        "avg_iters": float(iters.mean()),
        "p95_iters": float(np.percentile(iters, 95)),
        "max_iters": int(iters.max()),
        "runtime_s": float(elapsed),
        # final temps (for sanity / accuracy comparison)
        "T_sA_end": float(Y[-1, IDX["T_sA"]]),
        "T_sB_end": float(Y[-1, IDX["T_sB"]]),
        "T_cR_end": float(Y[-1, IDX["T_cR"]]),
        "T_r_end":  float(Y[-1, IDX["T_r"]]),
    }
    return t, Y, st, metrics


def simulate_fp(y0, t0, tf, dt, p, *, relax=0.2, max_iter=200, tol=1e-8):
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0.copy()

    iters = np.zeros(n_steps, dtype=int)
    ok = np.zeros(n_steps, dtype=bool)

    for n in range(n_steps):
        y_n = Y[n]
        t_n = t[n]
        y_np1, k, converged = step_implicit_euler_fixed_point(
            t_n, y_n, dt, p, max_iter=max_iter, tol=tol, relax=relax
        )
        Y[n + 1] = y_np1
        iters[n] = k
        ok[n] = converged

        if not np.all(np.isfinite(y_np1)):
            raise FloatingPointError(f"Non-finite state at step {n}, t={t_n}")

    return t, Y, {"iters": iters, "ok": ok}


def simulate_ngs(y0, t0, tf, dt, p, *, max_iter=50, tol=1e-10):
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0.copy()

    iters = np.zeros(n_steps, dtype=int)
    ok = np.zeros(n_steps, dtype=bool)

    for n in range(n_steps):
        y_n = Y[n]
        t_n = t[n]
        y_np1, k, converged = step_implicit_euler_newton_gs(
            t_n, y_n, dt, p, max_iter=max_iter, tol=tol
        )
        Y[n + 1] = y_np1
        iters[n] = k
        ok[n] = converged

        if not np.all(np.isfinite(y_np1)):
            raise FloatingPointError(f"Non-finite state at step {n}, t={t_n}")

    return t, Y, {"iters": iters, "ok": ok}


def compare_methods_dt_sweep(y0, p, *, t0=0.0, tf=2000.0, dts=(0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
                            fp_relax=0.2, fp_max_iter=300, ngs_max_iter=50):
    """
    Sweeps dt and returns:
      results: list of dict metrics (2 per dt)
      series:  dict keyed by (method, dt) -> (t, Y, st)
    """
    results = []
    series = {}

    for dt in dts:
        # Fixed point
        t_fp, Y_fp, st_fp, m_fp = run_one(
            "fp", y0, p, t0=t0, tf=tf, dt=dt,
            fp_relax=fp_relax, fp_max_iter=fp_max_iter,
        )
        results.append(m_fp)
        series[("fp", dt)] = (t_fp, Y_fp, st_fp)

        # NGS
        t_ngs, Y_ngs, st_ngs, m_ngs = run_one(
            "ngs", y0, p, t0=t0, tf=tf, dt=dt,
            ngs_max_iter=ngs_max_iter,
        )
        results.append(m_ngs)
        series[("ngs", dt)] = (t_ngs, Y_ngs, st_ngs)

        # quick accuracy check: compare final states
        # (not a "truth", but good to show FP vs NGS are solving same implicit step)
        diff = np.max(np.abs(Y_fp[-1] - Y_ngs[-1]))
        print(f"dt={dt:>6}  ok_fp={m_fp['ok_rate']:.2f}  ok_ngs={m_ngs['ok_rate']:.2f}  "
              f"avgIter_fp={m_fp['avg_iters']:.1f}  avgSweep_ngs={m_ngs['avg_iters']:.1f}  "
              f"final_inf_diff={diff:.3e}")

    return results, series


def plot_compare_time_series(t_fp, Y_fp, t_ngs, Y_ngs, *, title=""):
    """
    Overlay key temperatures for FP and NGS.
    """
    key = ["T_sA", "T_cA", "T_cR", "T_r"]
    plt.figure(figsize=(10, 5))
    for name in key:
        i = IDX[name]
        plt.plot(t_fp,  Y_fp[:, i],  label=f"{name} (FP)",  linestyle="-")
        plt.plot(t_ngs, Y_ngs[:, i], label=f"{name} (NGS)", linestyle="--")
    plt.xlabel("t (s)")
    plt.ylabel("Temperature (K)")
    plt.title(title or "Key temperatures: FP vs Newton–Gauss–Seidel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_iters_hist(st_fp, st_ngs, *, title=""):
    """
    Compare iteration counts distribution.
    """
    plt.figure(figsize=(9, 4))
    plt.hist(st_fp["iters"], bins=30, alpha=0.6, label="FP iters/step")
    plt.hist(st_ngs["iters"], bins=30, alpha=0.6, label="NGS sweeps/step")
    plt.xlabel("Iterations / sweeps per timestep")
    plt.ylabel("Count")
    plt.title(title or "Nonlinear solver effort per timestep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def results_to_markdown_table(results):
    # group by dt
    dts = sorted(set(r["dt"] for r in results))
    rows = []
    header = ("dt | FP ok% | FP avg it | FP p95 | FP time(s) | "
              "NGS ok% | NGS avg sw | NGS p95 | NGS time(s)")
    rows.append(header)
    rows.append("---|---:|---:|---:|---:|---:|---:|---:|---:")

    for dt in dts:
        r_fp = next(r for r in results if r["dt"] == dt and r["method"] == "fp")
        r_ng = next(r for r in results if r["dt"] == dt and r["method"] == "ngs")
        rows.append(
            f"{dt:g} | "
            f"{100*r_fp['ok_rate']:.1f} | {r_fp['avg_iters']:.1f} | {r_fp['p95_iters']:.0f} | {r_fp['runtime_s']:.2f} | "
            f"{100*r_ng['ok_rate']:.1f} | {r_ng['avg_iters']:.1f} | {r_ng['p95_iters']:.0f} | {r_ng['runtime_s']:.2f}"
        )
    return "\n".join(rows)

if __name__ == "__main__":
    sanity_check_finite_and_signs()

    p = default_params()

    # Balanced-ish scenario (so it doesn't drift forever):
    p["s_r"] = 0.025
    p["s_p"] = 0.0

    # Initial condition
    y0 = np.zeros(N_STATE)
    for name in STATE_NAMES:
        y0[IDX[name]] = 300.0

    # 1) Run a dt sweep and collect metrics
    dts = (0.5, 1.0, 2.0)
    results, series = compare_methods_dt_sweep(
        y0, p, t0=0.0, tf=500.0, dts=dts,
        fp_relax=0.2, fp_max_iter=300, ngs_max_iter=50
    )

    # 2) Print a ready-to-paste markdown table for the report
    print("\n=== Markdown table (paste into paper) ===\n")
    print(results_to_markdown_table(results))
    print("\n========================================\n")

    # 3) Pick ONE dt and show overlay plots + histogram
    dt_show = 2.0
    t_fp, Y_fp, st_fp = series[("fp", dt_show)]
    t_ng, Y_ng, st_ng = series[("ngs", dt_show)]

    plot_compare_time_series(
        t_fp, Y_fp, t_ng, Y_ng,
        title=f"FP vs Newton–GS (Implicit Euler), dt={dt_show}s"
    )
    plot_iters_hist(
        st_fp, st_ng,
        title=f"Nonlinear solver effort per timestep, dt={dt_show}s"
    )

    # 4) animate the NGS solution
    # pos, edges = make_layout()
    # animate_network(t_ng, Y_ng, pos, edges, stride=5, interval_ms=60)
