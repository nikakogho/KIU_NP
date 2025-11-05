# Finite Differences vs Exact Derivatives — Tangent Line/Plane and Normals

This project compares exact derivatives with finite-difference (FD) approximations for a 1D curve and a 2D surface. It builds tangent lines and planes, constructs normal vectors, and quantifies how FD step size affects accuracy using plots and tables. Everything runs locally and saves all evidence into a folder named fd_results.

---

## How to run

1. Install Python packages: numpy, pandas, matplotlib
2. Run from a terminal:
   python your_script_name.py
3. Results are written to: fd_results

Saved files:

* fd_results/table_1d_errors.csv
* fd_results/table_2d_errors.csv
* fd_results/fig_1d_tangents.png
* fd_results/fig_1d_derivative_error.png
* fd_results/fig_1d_normal_angle.png
* fd_results/fig_2d_surface_planes.png
* fd_results/fig_2d_gradient_error.png
* fd_results/fig_2d_normal_angle.png

---

## What we did

1. Chose two smooth test functions
   • Curve in 2D: f(x) = sin(x) · exp(−x²/10)
   • Surface in 3D: g(x, y) = x² y + sin(xy)

2. Computed exact derivatives (ground truth)
   • f′(x) = exp(−x²/10) · (cos x − (x/5) sin x)
   • ∇g(x, y) = (gₓ, gᵧ) with
   gₓ = 2xy + y cos(xy) and gᵧ = x² + x cos(xy)

3. Approximated derivatives numerically (no sklearn)
   At the point (x₀, y₀) = (0.7, −0.6) we used three standard finite-difference schemes over step sizes h ∈ [10⁻⁶, 10⁻¹]:
   • Forward difference (first order)
   • Central difference (second order)
   • Five-point central difference (fourth order)

For the surface, these stencils were applied along x and along y to approximate gₓ and gᵧ.

4. Built geometric objects from derivatives
   • Tangent line to y = f(x) at x₀: y = f(x₀) + f′(x₀)(x − x₀)
   • Curve normal (non-unit) at x₀: (−f′(x₀), 1)
   • Tangent plane to z = g(x, y) at (x₀, y₀):
   z = g(x₀, y₀) + gₓ(x₀, y₀)(x − x₀) + gᵧ(x₀, y₀)(y − y₀)
   • Surface normal (non-unit) at (x₀, y₀): (gₓ, gᵧ, −1)

Unit normals are obtained by normalizing these vectors.

5. Quantified accuracy with multiple metrics
   For each h and each scheme we computed:
   • 1D: absolute derivative error |f′(x₀) − FD|
   • 2D: gradient error ‖∇g(x₀, y₀) − FD‖₂
   • Angle error (in degrees) between exact and FD normals
   • Shape error of tangents (RMSE between the exact and FD tangent line over a local x-window)
   • Shape error of planes (RMSE between the exact and FD tangent plane over a local (x, y) patch)

6. Produced tables and plots as evidence
   All per-h results are stored in CSVs. Plots illustrate convergence and geometric accuracy.

---

## Functions and how they were handled

• f(x) captures oscillation with mild decay, ensuring nontrivial derivatives without singularities. Its exact derivative is available in closed form for clean error measurement.
• g(x, y) mixes a polynomial term and a sinusoidal coupling xy, producing gradients that vary in both directions. Exact partials gₓ and gᵧ are also closed form.

We evaluated everything at (x₀, y₀) = (0.7, −0.6). Local neighborhoods around this point were used to compare the shapes of the exact vs. FD tangent line and plane via RMSE.

---

## Finite-difference schemes used

| Scheme     | Stencil                                  | Expected order |
| ---------- | ---------------------------------------- | -------------- |
| Forward    | f(x), f(x + h)                           | O(h)           |
| Central    | f(x − h), f(x + h)                       | O(h²)          |
| Five-point | f(x − 2h), f(x − h), f(x + h), f(x + 2h) | O(h⁴)          |

The same idea is applied to g in x and y to get gₓ and gᵧ.

---

## What each figure shows

1. fd_results/fig_1d_tangents.png
   Displays f(x), the exact tangent line at x₀, and FD tangents built with representative step sizes for forward, central, and five-point schemes. This shows visually how the FD tangent approaches the exact tangent as the scheme and h improve.

2. fd_results/fig_1d_derivative_error.png
   Log-log plot of the absolute derivative error |f′(x₀) − FD| versus h. The lines exhibit the expected slopes for each method (approximately 1 for forward, 2 for central, 4 for five-point) until very small h where floating-point round-off limits further improvement.

3. fd_results/fig_1d_normal_angle.png
   Semilog plot of the angle between exact and FD curve normals versus h. Angle errors shrink rapidly for higher-order schemes at moderate h, confirming that improved derivative accuracy translates into more accurate geometry.

4. fd_results/fig_2d_surface_planes.png
   A 3D surface patch of z = g(x, y) around (x₀, y₀). Overlaid wireframes show the exact tangent plane, a coarse forward-difference plane (larger h), and a high-accuracy five-point plane (small h). Misalignment of the coarse plane is visually evident; the five-point plane aligns closely with the exact one.

5. fd_results/fig_2d_gradient_error.png
   Log-log plot of the gradient L2 error ‖∇g(x₀, y₀) − FD‖₂ versus h across schemes. As with the 1D case, the curves follow the expected orders until round-off becomes significant for tiny h.

6. fd_results/fig_2d_normal_angle.png
   Semilog plot of the angle between exact and FD surface normals versus h. Angle errors decrease markedly for higher-order schemes and suitable step sizes, reflecting improved tangent-plane orientation.

---

## Tables produced

• fd_results/table_1d_errors.csv
Columns: h, scheme, derivative_abs_error, normal_angle_deg, tangent_line_RMSE

• fd_results/table_2d_errors.csv
Columns: h, scheme, grad_l2_error, normal_angle_deg, plane_RMSE

These contain all measurements used to generate the figures and can be inspected or re-graphed independently.

---

## Results summary

• Forward, central, and five-point schemes demonstrate the expected convergence orders when h is not too small.
• Higher-order schemes produce dramatically smaller angle errors for normals and lower RMSE for tangent lines and planes.
• Very small step sizes eventually stop helping due to floating-point round-off, which is visible as the error curves flatten for tiny h.
