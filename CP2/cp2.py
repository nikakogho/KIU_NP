# cp2.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from model import IDX, N_STATE, STATE_NAMES, default_params, f, sanity_check_finite_and_signs


# ----------------------------
# Implicit Euler core
# ----------------------------

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


def step_implicit_euler_newton_gs(
    t_n, y_n, dt, p, *,
    max_sweeps=30, tol=1e-10, eps_fd=1e-6
):
    """
    Newton–Gauss–Seidel for g(y)=0.
    Sequential component update:
      y_i <- y_i - g_i(y) / (d g_i / d y_i)
    (diagonal derivative via finite difference)
    """
    t_np1 = t_n + dt
    y = y_n.copy()

    for sweep in range(max_sweeps):
        max_update = 0.0

        # compute residual at current y
        g = g_residual(t_np1, y, y_n, dt, p)

        for i in range(len(y)):
            g_i = g[i]

            # FD approx of diagonal derivative
            y_pert = y.copy()
            h = eps_fd * (1.0 + abs(y[i]))
            y_pert[i] += h

            g_pert = g_residual(t_np1, y_pert, y_n, dt, p)
            dg = (g_pert[i] - g_i) / h

            if abs(dg) < 1e-14:
                continue

            delta = -g_i / dg
            y[i] += delta
            max_update = max(max_update, abs(delta))

            # recompute residual after each update (more stable)
            g = g_residual(t_np1, y, y_n, dt, p)

            if not np.all(np.isfinite(y)):
                return y, sweep + 1, False

        if max_update < tol:
            return y, sweep + 1, True

    return y, max_sweeps, False


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
            y_np1, k, converged = step_implicit_euler_newton_gs(t_n, y_n, dt, p)
        else:
            raise ValueError("method must be 'fp' or 'ngs'")

        Y[n + 1] = y_np1
        iters[n] = k
        ok[n] = converged

        # if state blows up, stop early but keep array consistent
        if not np.all(np.isfinite(y_np1)):
            # fill remaining with last valid (or current) so plotting doesn't crash
            Y[n + 1 :] = y_np1
            iters[n + 1 :] = k
            ok[n + 1 :] = False
            break

    return t, Y, {"iters": iters, "ok": ok}


# ----------------------------
# Network layout + rendering
# ----------------------------

def make_layout():
    """
    Returns:
      pos: dict name -> (x,y)
      edges: list of (u, v, kind) where kind in {"pipe","link"}
    """
    pos = {}

    # GPU module nodes
    pos["T_sA"] = (-2.0,  0.6)
    pos["T_cA"] = (-1.3,  0.1)
    pos["T_sB"] = (-2.0, -0.6)
    pos["T_cB"] = (-1.3, -0.1)

    # Radiator nodes
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

    # Supply: radiator -> cold plates
    supA = arc_points(pos["T_cR"], pos["T_cA"], bulge=+0.9, n=5)
    supB = arc_points(pos["T_cR"], pos["T_cB"], bulge=-0.9, n=5)
    for j in range(1, 6):
        pos[f"T_sup_A{j}"] = supA[j - 1]
        pos[f"T_sup_B{j}"] = supB[j - 1]

    # Return: cold plates -> radiator
    retA = arc_points(pos["T_cA"], pos["T_cR"], bulge=-0.7, n=5)
    retB = arc_points(pos["T_cB"], pos["T_cR"], bulge=+0.7, n=5)
    for j in range(1, 6):
        pos[f"T_ret_A{j}"] = retA[j - 1]
        pos[f"T_ret_B{j}"] = retB[j - 1]

    edges = []

    # Branch A supply
    edges.append(("T_cR", "T_sup_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_A{j}", f"T_sup_A{j+1}", "pipe"))
    edges.append(("T_sup_A5", "T_cA", "pipe"))

    # Branch A return
    edges.append(("T_cA", "T_ret_A1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_A{j}", f"T_ret_A{j+1}", "pipe"))
    edges.append(("T_ret_A5", "T_cR", "pipe"))

    # Branch B supply
    edges.append(("T_cR", "T_sup_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_sup_B{j}", f"T_sup_B{j+1}", "pipe"))
    edges.append(("T_sup_B5", "T_cB", "pipe"))

    # Branch B return
    edges.append(("T_cB", "T_ret_B1", "pipe"))
    for j in range(1, 5):
        edges.append((f"T_ret_B{j}", f"T_ret_B{j+1}", "pipe"))
    edges.append(("T_ret_B5", "T_cR", "pipe"))

    # Non-pipe couplings
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
    """
    Create artists on the given Axes and return update(y, title_text) callable.
    """
    node_names = list(pos.keys())
    node_idx = np.array([IDX[n] for n in node_names], dtype=int)

    xs = np.array([pos[n][0] for n in node_names], dtype=float)
    ys = np.array([pos[n][1] for n in node_names], dtype=float)

    # Build edge segments and endpoint pairs
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

    # Optional boxes
    if show_boxes:
        ax.add_patch(Rectangle((-2.4, -1.05), 1.5, 2.1, fill=False, linewidth=2))
        ax.text(-2.35, 1.05, "GPU MODULE", fontsize=12, va="bottom")

        ax.add_patch(Rectangle((2.25, -0.55), 1.45, 1.1, fill=False, linewidth=2))
        ax.text(2.3, 0.6, "RADIATOR", fontsize=12, va="bottom")

    # Edge collections
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

    title = ax.set_title(title_text)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # breathing room
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    ax.set_xlim(xmin - 0.9, xmax + 0.9)
    ax.set_ylim(ymin - 0.9, ymax + 0.9)

    # Divergence label (hidden by default)
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
    suptitle="FP vs Newton–GS (Implicit Euler)"
):
    """
    Side-by-side animation with shared colormap.
    Robust vmin/vmax: computed ONLY from finite values.
    """
    # -------- robust color scale (finite-only) --------
    if scale_exclude_servers:
        scale_names = [n for n in pos.keys() if n not in ("T_sA", "T_sB")]
        scale_idx = np.array([IDX[n] for n in scale_names], dtype=int)
        A = np.concatenate([Y_fp[:, scale_idx], Y_ngs[:, scale_idx]], axis=0)
    else:
        A = np.concatenate([Y_fp, Y_ngs], axis=0)

    A = A.astype(float).ravel()
    A = A[np.isfinite(A)]

    if A.size == 0:
        # fallback (shouldn't happen)
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(A.min()), float(A.max())
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis  # keep consistent everywhere

    # -------- figure --------
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(suptitle, fontsize=16)

    # Build renderers (then set cmap consistently)
    update_fp = build_network_renderer(
        axs[0], pos, edges, norm, title_text="Fixed-Point"
    )
    update_ngs = build_network_renderer(
        axs[1], pos, edges, norm, title_text="Newton–Gauss–Seidel"
    )

    # IMPORTANT: apply cmap to all collections and scatters created on each Axes
    for ax in axs:
        for coll in ax.collections:
            try:
                coll.set_cmap(cmap)
            except Exception:
                pass

    # One shared colorbar
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

        update_fp(
            y_fp,
            title_override=f"Fixed-Point | t={t[i]:.1f}s",
            diverged=fp_div
        )
        update_ngs(
            y_ngs,
            title_override=f"Newton–GS | t={t[i]:.1f}s",
            diverged=ngs_div
        )
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



# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    sanity_check_finite_and_signs()

    p = default_params()

    # Initial condition: 300K everywhere
    y0 = np.full(N_STATE, 300.0, dtype=float)

    # For solver comparison, keep solar OFF (so stiffness/instability is from numerics, not external forcing)
    p["s_r"] = 0.0
    p["s_p"] = 0.0

    # Compare at a timestep that stresses FP (your results show dt=2.0 breaks FP)
    t0, tf, dt = 0.0, 500.0, 2.0

    t_fp, Y_fp, st_fp = simulate("fp", y0, t0=t0, tf=tf, dt=dt, p=p)
    t_ngs, Y_ngs, st_ngs = simulate("ngs", y0, t0=t0, tf=tf, dt=dt, p=p)

    print(f"FP:  ok.mean()={st_fp['ok'].mean():.3f}  avg iters={st_fp['iters'].mean():.2f}")
    print(f"NGS: ok.mean()={st_ngs['ok'].mean():.3f}  avg sweeps={st_ngs['iters'].mean():.2f}")

    pos, edges = make_layout()
    animate_compare(
        t_ngs, Y_fp, Y_ngs, pos, edges,
        stride=2,
        interval_ms=60,
        suptitle=f"FP vs Newton–GS (Implicit Euler), dt={dt:.1f}s"
    )
