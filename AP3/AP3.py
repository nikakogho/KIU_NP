# Finite Difference vs Exact Derivatives: Tangent line/plane & Normal vectors
# Saves CSVs/PNGs to ./fd_results

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Functions & exact derivatives

# 1D function
def f(x):
    return np.sin(x) * np.exp(-x**2/10.0)

def df_exact(x):
    # f'(x) = e^{-x^2/10} (cos x - (x/5) sin x)
    return np.exp(-x**2/10.0) * (np.cos(x) - (x/5.0)*np.sin(x))

# 2D function
def g(x, y):
    return x**2 * y + np.sin(x*y)

def gx_exact(x, y):
    return 2*x*y + np.cos(x*y)*y

def gy_exact(x, y):
    return x**2 + np.cos(x*y)*x

# 2) Finite-difference operators

# 1D derivatives
def d1_forward(fun, x, h):
    return (fun(x+h) - fun(x))/h

def d1_central(fun, x, h):
    return (fun(x+h) - fun(x-h))/(2*h)

def d1_five_point(fun, x, h):
    return (-fun(x+2*h) + 8*fun(x+h) - 8*fun(x-h) + fun(x-2*h)) / (12*h)

# 2D partials
def d2_partial_x(gfun, x, y, h, scheme="central"):
    if scheme == "forward":
        return (gfun(x+h, y) - gfun(x, y))/h
    elif scheme == "central":
        return (gfun(x+h, y) - gfun(x-h, y))/(2*h)
    elif scheme == "five_point":
        return (-gfun(x+2*h, y) + 8*gfun(x+h, y) - 8*gfun(x-h, y) + gfun(x-2*h, y))/(12*h)
    else:
        raise ValueError("Unknown scheme")

def d2_partial_y(gfun, x, y, h, scheme="central"):
    if scheme == "forward":
        return (gfun(x, y+h) - gfun(x, y))/h
    elif scheme == "central":
        return (gfun(x, y+h) - gfun(x, y-h))/(2*h)
    elif scheme == "five_point":
        return (-gfun(x, y+2*h) + 8*gfun(x, y+h) - 8*gfun(x, y-h) + gfun(x, y-2*h))/(12*h)
    else:
        raise ValueError("Unknown scheme")

# 3) Geometry helpers

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def angle_deg(u, v):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0: return np.nan
    c = np.dot(u, v) / (nu*nv)
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def tangent_line_at(x0, y0, slope):
    def line(x): return y0 + slope*(x - x0)
    return line

def tangent_plane_at(x0, y0, z0, gx, gy):
    def plane(x, y): return z0 + gx*(x - x0) + gy*(y - y0)
    return plane

# For surface z = g(x,y), a (non-unit) normal is n = (gx, gy, -1)
def surface_normal(gx, gy):
    return np.array([gx, gy, -1.0], dtype=float)

# For curve y = f(x), a (non-unit) normal in 2D is n = (-f'(x), 1)
def curve_normal_2d(df_val):
    return np.array([-df_val, 1.0], dtype=float)

# 4) Experiment setup & data collection

def run_experiment():
    x0, y0 = 0.7, -0.6

    f0 = f(x0)
    df0 = df_exact(x0)

    z0 = g(x0, y0)
    gx0 = gx_exact(x0, y0)
    gy0 = gy_exact(x0, y0)

    # Exact geometric objects
    line_exact  = tangent_line_at(x0, f0, df0)
    plane_exact = tangent_plane_at(x0, y0, z0, gx0, gy0)
    n_curve_exact = unit(curve_normal_2d(df0))
    n_surf_exact  = unit(surface_normal(gx0, gy0))

    # Step sizes and schemes
    hs = np.logspace(-6, -1, 12)  # 1e-6 ... 1e-1
    schemes_1d = [("forward", d1_forward), ("central", d1_central), ("five_point", d1_five_point)]
    schemes_2d = ["forward", "central", "five_point"]

    records_1d, records_2d = [], []

    # Local neighborhoods to compare tangent line/plane shapes (RMSE)
    xs_local = np.linspace(x0-0.5, x0+0.5, 200)
    X_local, Y_local = np.meshgrid(np.linspace(x0-0.4, x0+0.4, 60),
                                   np.linspace(y0-0.4, y0+0.4, 60))

    for h in hs:
        # 1D
        for name, op in schemes_1d:
            df_hat = op(f, x0, h)
            derr = abs(df_hat - df0)
            n_hat = unit(curve_normal_2d(df_hat))
            ang = angle_deg(n_hat, n_curve_exact)
            line_hat = tangent_line_at(x0, f0, df_hat)
            rmse = float(np.sqrt(np.mean((line_exact(xs_local) - line_hat(xs_local))**2)))
            records_1d.append({"h": h, "scheme": name,
                               "derivative_abs_error": derr,
                               "normal_angle_deg": ang,
                               "tangent_line_RMSE": rmse})

        # 2D
        for scheme in schemes_2d:
            gx_hat = d2_partial_x(g, x0, y0, h, scheme=scheme)
            gy_hat = d2_partial_y(g, x0, y0, h, scheme=scheme)
            grad_err = float(np.sqrt((gx_hat-gx0)**2 + (gy_hat-gy0)**2))
            n_hat = unit(surface_normal(gx_hat, gy_hat))
            ang = angle_deg(n_hat, n_surf_exact)
            plane_hat = tangent_plane_at(x0, y0, z0, gx_hat, gy_hat)
            rmse = float(np.sqrt(np.mean((plane_exact(X_local, Y_local) - plane_hat(X_local, Y_local))**2)))
            records_2d.append({"h": h, "scheme": scheme,
                               "grad_l2_error": grad_err,
                               "normal_angle_deg": ang,
                               "plane_RMSE": rmse})

    df1 = pd.DataFrame(records_1d).sort_values(["scheme", "h"])
    df2 = pd.DataFrame(records_2d).sort_values(["scheme", "h"])
    return (x0, y0, f0, z0, df1, df2, line_exact, plane_exact)

# 5) Utility: estimate observed order (slope on log-log)
def estimate_orders(df, err_col, group_col="scheme", h_col="h",
                    h_lo=1e-5, h_hi=1e-2):
    orders = {}
    for key, sub in df.groupby(group_col):
        sub = sub[(sub[h_col] >= h_lo) & (sub[h_col] <= h_hi)]
        if len(sub) >= 2 and np.all(sub[err_col] > 0):
            x = np.log(sub[h_col].values)
            y = np.log(sub[err_col].values)
            p = np.polyfit(x, y, deg=1)  # y ~ p[0]*x + p[1]
            orders[key] = p[0]  # slope ~ order
    return orders

# 6) Plotting & saving
def make_and_save_plots(out_dir, x0, y0, f0, z0, df1, df2, line_exact, plane_exact):
    os.makedirs(out_dir, exist_ok=True)

    # 1D: function & tangents
    x_plot = np.linspace(x0-2, x0+2, 400)
    y_plot = f(x_plot)
    plt.figure()
    plt.plot(x_plot, y_plot, label="f(x)")
    plt.plot(x_plot, line_exact(x_plot), linestyle="--", label="Exact tangent")

    hs_show = [1e-1, 1e-3, 1e-5]
    df_forward_h1 = d1_forward(f, x0, hs_show[0])
    plt.plot(x_plot, tangent_line_at(x0, f0, df_forward_h1)(x_plot), linestyle=":", label="Forward h=1e-1")
    df_central_h2 = d1_central(f, x0, hs_show[1])
    plt.plot(x_plot, tangent_line_at(x0, f0, df_central_h2)(x_plot), linestyle="-.", label="Central h=1e-3")
    df_five_h3 = d1_five_point(f, x0, hs_show[2])
    plt.plot(x_plot, tangent_line_at(x0, f0, df_five_h3)(x_plot), linestyle="-", label="Five-point h=1e-5")
    plt.scatter([x0], [f0], marker="o", label="(x0, f(x0))")
    plt.title("1D: f(x) and tangent lines at x0")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_1d_tangents.png"), dpi=180)
    plt.close()

    # 1D: derivative error vs h
    plt.figure()
    for name, sub in df1.groupby("scheme"):
        plt.loglog(sub["h"], sub["derivative_abs_error"], marker="o", label=name)
    plt.title("1D: |f'(x0) - FD| vs step h")
    plt.xlabel("h (log)"); plt.ylabel("abs derivative error (log)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_1d_derivative_error.png"), dpi=180)
    plt.close()

    # 1D: normal angle vs h
    plt.figure()
    for name, sub in df1.groupby("scheme"):
        plt.semilogx(sub["h"], sub["normal_angle_deg"], marker="o", label=name)
    plt.title("1D: Normal angle error (deg) vs h")
    plt.xlabel("h (log)"); plt.ylabel("angle error (deg)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_1d_normal_angle.png"), dpi=180)
    plt.close()

    # 2D: surface & planes
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(x0-0.7, x0+0.7, 50)
    Y = np.linspace(y0-0.7, y0+0.7, 50)
    XX, YY = np.meshgrid(X, Y)
    ZZ = g(XX, YY)
    ax.plot_surface(XX, YY, ZZ, alpha=0.5)

    ZZ_plane_exact = plane_exact(XX, YY)
    ax.plot_wireframe(XX, YY, ZZ_plane_exact, rstride=5, cstride=5)

    gx_f = d2_partial_x(g, x0, y0, 1e-1, scheme="forward")
    gy_f = d2_partial_y(g, x0, y0, 1e-1, scheme="forward")
    plane_f = tangent_plane_at(x0, y0, z0, gx_f, gy_f)
    ax.plot_wireframe(XX, YY, plane_f(XX, YY), rstride=5, cstride=5)

    gx_5 = d2_partial_x(g, x0, y0, 1e-4, scheme="five_point")
    gy_5 = d2_partial_y(g, x0, y0, 1e-4, scheme="five_point")
    plane_5 = tangent_plane_at(x0, y0, z0, gx_5, gy_5)
    ax.plot_wireframe(XX, YY, plane_5(XX, YY), rstride=5, cstride=5)

    ax.set_title("2D: Surface and tangent planes at (x0,y0)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_2d_surface_planes.png"), dpi=180)
    plt.close()

    # 2D: gradient error vs h
    plt.figure()
    for name, sub in df2.groupby("scheme"):
        plt.loglog(sub["h"], sub["grad_l2_error"], marker="o", label=name)
    plt.title("2D: ||∇g(x0,y0) - FD|| vs h")
    plt.xlabel("h (log)"); plt.ylabel("gradient L2 error (log)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_2d_gradient_error.png"), dpi=180)
    plt.close()

    # 2D: normal angle vs h
    plt.figure()
    for name, sub in df2.groupby("scheme"):
        plt.semilogx(sub["h"], sub["normal_angle_deg"], marker="o", label=name)
    plt.title("2D: Normal angle error (deg) vs h")
    plt.xlabel("h (log)"); plt.ylabel("angle error (deg)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_2d_normal_angle.png"), dpi=180)
    plt.close()

# 7) Main
def main():
    out_dir = "fd_results"
    x0, y0, f0, z0, df1, df2, line_exact, plane_exact = run_experiment()

    # Print and save tables
    print("\n== 1D Finite Difference Errors ==")
    print(df1.to_string(index=False))
    print("\n== 2D Finite Difference Errors ==")
    print(df2.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    df1.to_csv(os.path.join(out_dir, "table_1d_errors.csv"), index=False)
    df2.to_csv(os.path.join(out_dir, "table_2d_errors.csv"), index=False)

    # Estimate observed orders (slopes on log-log in a mid-range to avoid round-off)
    orders_1d = estimate_orders(df1, err_col="derivative_abs_error")
    orders_2d = estimate_orders(df2, err_col="grad_l2_error")

    print("\nObserved order (1D derivative abs error ~ h^p) in mid-range:")
    for k, v in orders_1d.items():
        print(f"  {k:10s}: p ~ {v:.2f}")
    print("\nObserved order (2D gradient L2 error ~ h^p) in mid-range:")
    for k, v in orders_2d.items():
        print(f"  {k:10s}: p ~ {v:.2f}")

    # Plots
    make_and_save_plots(out_dir, x0, y0, f0, z0, df1, df2, line_exact, plane_exact)
    print(f"\nSaved tables and figures in: ./{out_dir}/")

if __name__ == "__main__":
    main()
