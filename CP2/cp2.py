# cp2.py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from model import IDX, N_STATE, STATE_NAMES, default_params, f, sanity_check_finite_and_signs


#
# Implicit Euler core
#

def g_residual(t_np1, y_guess, y_n, dt, p):
    """Implicit Euler residual: g(y) = y - y_n - dt * f(t_{n+1}, y)."""
    return y_guess - y_n - dt * f(t_np1, y_guess, p)


def step_implicit_euler_fixed_point(
    t_n, y_n, dt, p, *,
    max_iter=200, tol=1e-8, relax=0.2
):
    """
    Fixed-point iterations for implicit Euler:
      y^{k+1} = y_n + dt * f(t_{n+1}, y^k)
    Optional relaxation:
      y^{k+1} <- (1-relax)*y^k + relax*y^{k+1}
    """
    t_np1 = t_n + dt
    y = y_n.copy()

    for k in range(max_iter):
        y_new = y_n + dt * f(t_np1, y, p)
        if relax != 1.0:
            y_new = (1.0 - relax) * y + relax * y_new

        if not np.all(np.isfinite(y_new)):
            return y_new, k + 1, False

        err = np.max(np.abs(y_new - y))
        y = y_new
        if err < tol:
            return y, k + 1, True

    return y, max_iter, False


def step_implicit_euler_newton_dense(
    t_n, y_n, dt, p, *,
    max_iter=12,
    tol_g=1e-6,
    tol_update=1e-8,
    eps_fd=1e-6,
    damping_min=1e-3,
    max_step_norm=5e3,
):
    """
    Newton solve for implicit Euler: g(y)=0 where
      g(y) = y - y_n - dt*f(t_{n+1}, y)

    Finite-difference dense Jacobian + linear solve + backtracking line search.

    Returns (y_np1, iters, converged).
    """
    t_np1 = t_n + dt
    y = y_n.copy().astype(float)

    def g(y_local):
        return y_local - y_n - dt * f(t_np1, y_local, p)

    g0 = g(y)
    if not np.all(np.isfinite(g0)):
        return y, 0, False

    gnorm0 = float(np.linalg.norm(g0, ord=np.inf))
    if gnorm0 < tol_g:
        return y, 0, True

    n = len(y)

    for it in range(1, max_iter + 1):
        J = np.zeros((n, n), dtype=float)
        g_base = g0

        for j in range(n):
            yj = y[j]
            h = eps_fd * (1.0 + abs(yj))
            y_pert = y.copy()
            y_pert[j] = yj + h

            g_pert = g(y_pert)
            if not np.all(np.isfinite(g_pert)):
                return y, it, False

            J[:, j] = (g_pert - g_base) / h

        try:
            delta = np.linalg.solve(J, -g_base)
        except np.linalg.LinAlgError:
            return y, it, False

        if not np.all(np.isfinite(delta)):
            return y, it, False

        step_norm = float(np.linalg.norm(delta, ord=np.inf))
        if step_norm > max_step_norm:
            delta *= (max_step_norm / step_norm)

        alpha = 1.0
        accepted = False
        gnorm = float(np.linalg.norm(g_base, ord=np.inf))

        while alpha >= damping_min:
            y_try = y + alpha * delta
            g_try = g(y_try)
            if np.all(np.isfinite(g_try)):
                gnorm_try = float(np.linalg.norm(g_try, ord=np.inf))
                if gnorm_try < (1.0 - 1e-4 * alpha) * gnorm:
                    y = y_try
                    g0 = g_try
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            return y, it, False

        gnorm_new = float(np.linalg.norm(g0, ord=np.inf))
        upd_norm = float(np.linalg.norm(alpha * delta, ord=np.inf))

        if gnorm_new < tol_g or upd_norm < tol_update:
            return y, it, True

    return y, max_iter, False


def simulate(method, y0, t0, tf, dt, p):
    """
    method: "fp" or "ngs"
    Returns:
      t: (N,) times
      Y: (N, N_STATE)
      stats: dict with iters and ok flags
    """
    n_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)), dtype=float)
    Y[0] = y0.copy()

    iters = np.zeros(n_steps, dtype=int)
    ok = np.zeros(n_steps, dtype=bool)

    for n in range(n_steps):
        y_n = Y[n]
        t_n = t[n]

        if method == "fp":
            y_np1, k, converged = step_implicit_euler_fixed_point(t_n, y_n, dt, p)
        elif method == "ngs":
            y_np1, k, converged = step_implicit_euler_newton_dense(t_n, y_n, dt, p)
        else:
            raise ValueError("method must be 'fp' or 'ngs'")

        Y[n + 1] = y_np1
        iters[n] = k
        ok[n] = converged

        if not np.all(np.isfinite(y_np1)):
            Y[n + 1:] = y_np1
            iters[n + 1:] = k
            ok[n + 1:] = False
            break

    return t, Y, {"iters": iters, "ok": ok}


#
# Network layout + rendering
#

def make_layout():
    """
    Returns:
      pos: dict name -> (x,y)
      edges: list of (u, v, kind) where kind in {"pipe","link"}
    """
    pos = {}

    pos["T_sA"] = (-2.0,  0.6)
    pos["T_cA"] = (-1.3,  0.1)
    pos["T_sB"] = (-2.0, -0.6)
    pos["T_cB"] = (-1.3, -0.1)

    pos["T_cR"] = ( 2.6,  0.0)
    pos["T_r"]  = ( 3.4,  0.0)

    def arc_points(start, end, bulge, n=5):
        x0, y0 = start
        x1, y1 = end
        pts = []
        for j in range(1, n + 1):
            s = j / (n + 1)
            x = (1 - s) * x0 + s * x1
            y = (1 - s) * y0 + s * y1
            y += bulge * np.sin(np.pi * s)
            pts.append((x, y))
        return pts

    supA = arc_points(pos["T_cR"], pos["T_cA"], bulge=+0.9, n=5)
    supB = arc_points(pos["T_cR"], pos["T_cB"], bulge=-0.9, n=5)
    for j in range(1, 6):
        pos[f"T_sup_A{j}"] = supA[j - 1]
        pos[f"T_sup_B{j}"] = supB[j - 1]

    retA = arc_points(pos["T_cA"], pos["T_cR"], bulge=-0.7, n=5)
    retB = arc_points(pos["T_cB"], pos["T_cR"], bulge=+0.7, n=5)
    for j in range(1, 6):
        pos[f"T_ret_A{j}"] = retA[j - 1]
        pos[f"T_ret_B{j}"] = retB[j - 1]

    edges = []

    edges.append(("T_cR", "T_sup_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_A{j}", f"T_sup_A{j+1}", "pipe"))
    edges.append(("T_sup_A5", "T_cA", "pipe"))

    edges.append(("T_cA", "T_ret_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_A{j}", f"T_ret_A{j+1}", "pipe"))
    edges.append(("T_ret_A5", "T_cR", "pipe"))

    edges.append(("T_cR", "T_sup_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_B{j}", f"T_sup_B{j+1}", "pipe"))
    edges.append(("T_sup_B5", "T_cB", "pipe"))

    edges.append(("T_cB", "T_ret_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_B{j}", f"T_ret_B{j+1}", "pipe"))
    edges.append(("T_ret_B5", "T_cR", "pipe"))

    edges.append(("T_sA", "T_cA", "link"))
    edges.append(("T_sB", "T_cB", "link"))
    edges.append(("T_cR", "T_r", "link"))

    return pos, edges


def build_network_renderer(
    ax, pos, edges, norm, *,
    title_text="Thermal snapshot",
    show_labels=True,
    label_fontsize=8,
    label_offset=(0.05, 0.05),
    show_boxes=True
):
    node_names = list(pos.keys())
    node_idx = np.array([IDX[n] for n in node_names], dtype=int)

    xs = np.array([pos[n][0] for n in node_names], dtype=float)
    ys = np.array([pos[n][1] for n in node_names], dtype=float)

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

    if show_boxes:
        ax.add_patch(Rectangle((-2.4, -1.05), 1.5, 2.1, fill=False, linewidth=2))
        ax.text(-2.35, 1.05, "GPU MODULE", fontsize=12, va="bottom")

        ax.add_patch(Rectangle((2.25, -0.55), 1.45, 1.1, fill=False, linewidth=2))
        ax.text(2.3, 0.6, "RADIATOR", fontsize=12, va="bottom")

    lc_pipe = LineCollection(pipe_segs, linewidths=6)
    lc_pipe.set_norm(norm)
    ax.add_collection(lc_pipe)

    lc_link = LineCollection(link_segs, linewidths=2, linestyles="dashed")
    lc_link.set_norm(norm)
    ax.add_collection(lc_link)

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

    title = ax.set_title(title_text)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    ax.set_xlim(xmin - 0.9, xmax + 0.9)
    ax.set_ylim(ymin - 0.9, ymax + 0.9)

    diverge_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="red", alpha=0.9),
        color="red",
        zorder=10,
    )

    def update_with_y(y, *, title_override=None, diverged=False):
        if len(pipe_pairs):
            pipe_vals = 0.5 * (y[pipe_pairs[:, 0]] + y[pipe_pairs[:, 1]])
            lc_pipe.set_array(pipe_vals)
        if len(link_pairs):
            link_vals = 0.5 * (y[link_pairs[:, 0]] + y[link_pairs[:, 1]])
            lc_link.set_array(link_vals)

        sc.set_array(y[node_idx])

        if title_override is not None:
            title.set_text(title_override)

        diverge_text.set_text("DIVERGED" if diverged else "")

        return lc_pipe, lc_link, sc, title, diverge_text

    return update_with_y


def animate_compare(
    t, Y_fp, Y_ngs, pos, edges, *,
    stride=5,
    interval_ms=60,
    scale_exclude_servers=True,
    suptitle="FP vs Damped Newton (finite-difference dense Jacobian)"
):
    if scale_exclude_servers:
        scale_names = [n for n in pos.keys() if n not in ("T_sA", "T_sB")]
        scale_idx = np.array([IDX[n] for n in scale_names], dtype=int)
        A = np.concatenate([Y_fp[:, scale_idx], Y_ngs[:, scale_idx]], axis=0)
    else:
        A = np.concatenate([Y_fp, Y_ngs], axis=0)

    A = A.astype(float).ravel()
    A = A[np.isfinite(A)]

    if A.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(A.min()), float(A.max())
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(suptitle, fontsize=16)

    update_fp = build_network_renderer(axs[0], pos, edges, norm, title_text="Fixed-Point")
    update_ngs = build_network_renderer(axs[1], pos, edges, norm, title_text="Damped Newton")

    for ax in axs:
        for coll in ax.collections:
            try:
                coll.set_cmap(cmap)
            except Exception:
                pass

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label("Temperature (K)")

    frame_indices = np.arange(0, len(t), stride, dtype=int)

    def safe_row(Y, i):
        if np.all(np.isfinite(Y[i])):
            return Y[i], False
        j = i
        while j > 0 and not np.all(np.isfinite(Y[j])):
            j -= 1
        return Y[j], True

    def anim_update(k):
        i = frame_indices[k]
        y_fp, fp_div = safe_row(Y_fp, i)
        y_ngs, ngs_div = safe_row(Y_ngs, i)

        update_fp(y_fp, title_override=f"Fixed-Point | t={t[i]:.1f}s", diverged=fp_div)
        update_ngs(y_ngs, title_override=f"Damped Newton | t={t[i]:.1f}s", diverged=ngs_div)
        return []

    anim = FuncAnimation(
        fig, anim_update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,
        repeat=True
    )
    plt.show()
    return anim


#
# Minimal comparator suite
#

def _select_state_indices(exclude_servers=True):
    if not exclude_servers:
        return np.arange(N_STATE, dtype=int)
    bad = {IDX["T_sA"], IDX["T_sB"]}
    idx = [i for i in range(N_STATE) if i not in bad]
    return np.array(idx, dtype=int)


def _max_rms_error_vs_ref(Y, Yref, stride, idx_use, *, min_points=10, min_frac=0.5):
    Yref_s = Yref[::stride, :]
    n = min(len(Y), len(Yref_s))
    A = Y[:n, idx_use]
    B = Yref_s[:n, idx_use]

    ok = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    used = int(np.sum(ok))
    frac = used / max(1, n)

    if used < min_points or frac < min_frac:
        return np.nan, frac

    diff = A[ok] - B[ok]
    rms_t = np.sqrt(np.mean(diff**2, axis=1))
    return float(np.max(rms_t)), frac


def plot_with_divergence_markers(x, y, diverged, *, label, marker="o"):
    plt.plot(x, y, marker=marker, label=label)

    finite = np.isfinite(y)
    if np.any(finite):
        ymax = float(np.max(y[finite]))
        y_mark = ymax * 1.10 if ymax != 0 else 1.0
    else:
        y_mark = 1.0

    if np.any(diverged):
        plt.scatter(
            x[diverged],
            np.full(np.sum(diverged), y_mark),
            marker="x",
            s=100,
            label=f"{label} diverged",
            zorder=10,
        )
    return y_mark


def run_comparison_suite(
    *,
    tf=500.0,
    dt_ref=0.25,
    dts=(0.5, 1.0, 2.0, 4.0),
    exclude_servers=True,
    make_plots=True
):
    sanity_check_finite_and_signs()
    p = default_params()
    p["s_r"] = 0.0
    p["s_p"] = 0.0

    y0 = np.full(N_STATE, 300.0, dtype=float)
    t0 = 0.0

    idx_use = _select_state_indices(exclude_servers=exclude_servers)

    print(f"\n[Reference] DN with dt_ref={dt_ref}, tf={tf} ...")
    t_start = time.perf_counter()
    t_ref, Y_ref, st_ref = simulate("ngs", y0, t0=t0, tf=tf, dt=dt_ref, p=p)
    ref_time = time.perf_counter() - t_start
    ref_ok_rate = float(np.mean(st_ref["ok"])) if len(st_ref["ok"]) else 0.0
    print(f"  ref runtime={ref_time:.3f}s, ok.mean={ref_ok_rate:.3f}, avg iters={st_ref['iters'].mean():.2f}")

    rows = []
    for dt in dts:
        stride = dt / dt_ref
        if abs(stride - round(stride)) > 1e-12:
            raise ValueError(f"dt={dt} must be a multiple of dt_ref={dt_ref} (got stride={stride}).")
        stride = int(round(stride))

        for method in ("fp", "ngs"):
            print(f"\n[{method.upper()}] dt={dt} (stride={stride}) ...")
            t_start = time.perf_counter()
            t_m, Y_m, st_m = simulate(method, y0, t0=t0, tf=tf, dt=dt, p=p)
            runtime = time.perf_counter() - t_start

            ok_rate = float(np.mean(st_m["ok"])) if len(st_m["ok"]) else 0.0
            avg_iter = float(np.mean(st_m["iters"])) if len(st_m["iters"]) else np.nan

            err, used_frac = _max_rms_error_vs_ref(Y_m, Y_ref, stride=stride, idx_use=idx_use)

            traj_has_nan = not np.all(np.isfinite(Y_m))
            diverged = traj_has_nan or (not np.isfinite(err)) or (ok_rate < 0.99)

            rows.append({
                "method": method,
                "dt": float(dt),
                "runtime_s": float(runtime),
                "ok_rate": ok_rate,
                "avg_iters": avg_iter,
                "max_rms_err": float(err) if np.isfinite(err) else np.nan,
                "used_frac": float(used_frac),
                "diverged": bool(diverged),
            })

            print(
                f"  runtime={runtime:.3f}s | ok.mean={ok_rate:.3f} | avg iters={avg_iter:.2f} | "
                f"used_frac={used_frac:.2f} | max RMS err={err}"
            )

    rows_fp = [r for r in rows if r["method"] == "fp"]
    rows_ngs = [r for r in rows if r["method"] == "ngs"]

    x_fp = np.array([r["dt"] for r in rows_fp], dtype=float)
    x_ngs = np.array([r["dt"] for r in rows_ngs], dtype=float)

    div_fp = np.array([r["diverged"] for r in rows_fp], dtype=bool)
    div_ngs = np.array([r["diverged"] for r in rows_ngs], dtype=bool)

    err_fp = np.array([r["max_rms_err"] for r in rows_fp], dtype=float)
    err_ngs = np.array([r["max_rms_err"] for r in rows_ngs], dtype=float)

    it_fp = np.array([r["avg_iters"] for r in rows_fp], dtype=float)
    it_ngs = np.array([r["avg_iters"] for r in rows_ngs], dtype=float)

    ok_fp = np.array([r["ok_rate"] for r in rows_fp], dtype=float)
    ok_ngs = np.array([r["ok_rate"] for r in rows_ngs], dtype=float)

    rt_fp = np.array([r["runtime_s"] for r in rows_fp], dtype=float)
    rt_ngs = np.array([r["runtime_s"] for r in rows_ngs], dtype=float)

    err_fp_plot = err_fp.copy()
    err_ngs_plot = err_ngs.copy()
    err_fp_plot[div_fp] = np.nan
    err_ngs_plot[div_ngs] = np.nan

    it_fp_plot = it_fp.copy()
    it_ngs_plot = it_ngs.copy()
    it_fp_plot[div_fp] = np.nan
    it_ngs_plot[div_ngs] = np.nan

    if make_plots:
        plt.figure(figsize=(8, 5))
        plot_with_divergence_markers(x_fp, err_fp_plot, div_fp, label="Fixed-Point")
        plot_with_divergence_markers(x_ngs, err_ngs_plot, div_ngs, label="Damped Newton")
        plt.xlabel("dt (s)")
        plt.ylabel("Max RMS error vs reference (K)")
        plt.title("Accuracy vs timestep (divergence-aware)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(bottom=0)
        plt.show()

        plt.figure(figsize=(8, 5))
        plot_with_divergence_markers(x_fp, it_fp_plot, div_fp, label="Fixed-Point")
        plot_with_divergence_markers(x_ngs, it_ngs_plot, div_ngs, label="Damped Newton")
        plt.xlabel("dt (s)")
        plt.ylabel("Average iterations / sweeps per step")
        plt.title("Work per step vs timestep (divergence-aware)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(bottom=0)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(x_fp, ok_fp, marker="o", label="Fixed-Point")
        plt.plot(x_ngs, ok_ngs, marker="o", label="Damped Newton")
        if np.any(div_fp):
            plt.scatter(x_fp[div_fp], np.full(np.sum(div_fp), 1.02), marker="x", s=100, label="FP diverged", zorder=10)
        if np.any(div_ngs):
            plt.scatter(x_ngs[div_ngs], np.full(np.sum(div_ngs), 1.02), marker="x", s=100, label="DN diverged", zorder=10)
        plt.xlabel("dt (s)")
        plt.ylabel("ok.mean() (fraction converged steps)")
        plt.title("Robustness vs timestep (divergence-aware)")
        plt.ylim(-0.05, 1.10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(x_fp, rt_fp, marker="o", label="Fixed-Point")
        plt.plot(x_ngs, rt_ngs, marker="o", label="Damped Newton")
        if np.any(div_fp):
            plt.scatter(x_fp[div_fp], rt_fp[div_fp], marker="x", s=100, label="FP diverged", zorder=10)
        if np.any(div_ngs):
            plt.scatter(x_ngs[div_ngs], rt_ngs[div_ngs], marker="x", s=100, label="DN diverged", zorder=10)
        plt.xlabel("dt (s)")
        plt.ylabel("Runtime (seconds)")
        plt.title("Runtime vs timestep (divergence-aware)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(bottom=0)
        plt.show()

    print("\n=== Summary ===")
    print("method   dt    runtime_s   ok.mean   avg_iters   max_rms_err   diverged")
    for r in rows:
        print(
            f"{r['method']:>6}  {r['dt']:>4.2f}   {r['runtime_s']:>8.3f}   "
            f"{r['ok_rate']:>6.3f}    {r['avg_iters']:>8.2f}   "
            f"{r['max_rms_err']!s:>10}   {str(r['diverged']):>8}"
        )

    return rows


#
# Main
#

if __name__ == "__main__":
    sanity_check_finite_and_signs()

    p = default_params()
    y0 = np.full(N_STATE, 300.0, dtype=float)
    p["s_r"] = 0.0
    p["s_p"] = 0.0

    t0, tf, dt = 0.0, 500.0, 2.0
    t_fp, Y_fp, st_fp = simulate("fp", y0, t0=t0, tf=tf, dt=dt, p=p)
    t_ngs, Y_ngs, st_ngs = simulate("ngs", y0, t0=t0, tf=tf, dt=dt, p=p)

    print(f"FP:  ok.mean()={st_fp['ok'].mean():.3f}  avg iters={st_fp['iters'].mean():.2f}")
    print(f"DN: ok.mean()={st_ngs['ok'].mean():.3f}  avg iters={st_ngs['iters'].mean():.2f}")

    pos, edges = make_layout()
    animate_compare(
        t_ngs, Y_fp, Y_ngs, pos, edges,
        stride=2,
        interval_ms=60,
        suptitle=f"FP vs Damped Newton, dt={dt:.1f}s"
    )

    run_comparison_suite(
        tf=500.0,
        dt_ref=0.5,
        dts=(0.5, 1.0, 2.0, 4.0),
        exclude_servers=True,
        make_plots=True
    )
