# Robot Navigation Simulation (Numerical Programming Project)

This repo solves a 3-part robot navigation assignment using **image-based corridor extraction**, **spline path parametrization**, and a **physics-inspired IVP** (initial value problem) integrated with **semi-implicit Euler**.

You will find:

* **Problem 1:** single robot follows a spline centerline inside a corridor
* **Problem 2:** two swarms travel **A→B** and **B→A** simultaneously without collisions
* **Problem 3:** robot navigates through **pedestrian flow** (synthetic GT or real video), in **two directions**

---

## Problem statement (assignment)

1. **Follow a path in a constrained environment**
   Input: map image + points A and B connected by a path, path width, robot size
   Task: extract the A→B path and parametrize using splines; robot moves A→B without crossing borders
   Output: path (with width) + visualization of robot following it

2. **Swarm of robots in a constrained path**
   Input: a path with width + swarms at A and at B
   Task: swarms autonomously navigate A→B and B→A simultaneously without accidents
   Output: visualization of robots navigating without collisions

3. **Navigation in a pedestrian flow**
   Input: pedestrian flow video
   Task: robot safely navigates in at least two directions
   Output: visualization including robot + pedestrians

---

## Core model used everywhere (IVP + semi-implicit Euler)

Each robot has position $x \in \mathbb{R}^2$ and velocity $v \in \mathbb{R}^2$. We use a target-tracking + damping + repulsion model:

$$
a(x,v) = \frac{1}{m}\Big(k_p(T-x) - k_d v + f_{\text{rep}}(x) + f_{\text{ext}}(x)\Big)
$$

Semi-implicit Euler step (stable for “mass-spring-damper”-style motion):

$$
v_{n+1} = \text{sat}\big(v_n + \Delta t , a(x_n, v_n)\big), \qquad
x_{n+1} = x_n + \Delta t , v_{n+1}
$$

Velocity saturation (enforces $\lVert v \rVert \le v_{\max}$):

$$
\text{sat}(v)=
\begin{cases}
v & \lVert v\rVert \le v_{\max}\
v\cdot \frac{v_{\max}}{\lVert v\rVert} & \text{otherwise}
\end{cases}
$$

### Pairwise robot–robot repulsion (Problem 2)

For robots $i \ne j$, within a safety radius $R_{\text{safe}}$:

$$
f_{ij} = k_{\text{rep}}\frac{x_i-x_j}{\lVert x_i-x_j\rVert^3}, \quad \text{if } \lVert x_i-x_j\rVert < R_{\text{safe}}
$$

Total repulsion on robot $i$ is $f_{\text{rep},i}=\sum_{j\ne i} f_{ij}$.

### Wall repulsion via distance transform gradient (Problems 1–3)

We compute a **safe corridor mask** (eroded by robot radius), then compute a distance-to-wall field $d(x)$ using a distance transform. Its gradient points inward (toward safer interior).

If $d(x) < \text{margin}$, apply:

$$
f_{\text{wall}}(x)=k_{\text{wall}}\big(\text{margin}-d(x)\big),\widehat{\nabla d(x)}
$$

where $\widehat{\nabla d(x)}=\frac{\nabla d(x)}{\lVert \nabla d(x)\rVert+\epsilon}$.

---

## Map preprocessing (corridor mask)

Given a BGR image:

1. Convert to grayscale + optional Gaussian blur
2. Threshold using Otsu (auto-tries normal vs inverted)
3. Morphological close/open cleanup
4. Optionally keep the **largest connected component**

Output is a binary mask `mask255` with values `{0,255}` representing the corridor.

Files:

* `navigation/preprocess.py` (`path_mask_from_bgr`, cleanup utilities)

---

## Path extraction and spline parametrization

### A→B centerline pixel path (BFS or Dijkstra)

Two options exist:

* **BFS shortest path** on corridor pixels (baseline)
* **Dijkstra with center preference** (used for final centerline)

Clearance $c$ is computed via distance transform on the corridor mask (bigger = farther from boundary).
For each step to a neighbor pixel, the cost is:

$$
\text{cost} = \ell \left(1 + \frac{\lambda}{c+\epsilon}\right)
$$

where $\ell$ is step length (1 or $\sqrt{2}$), and $\lambda$ is `center_weight`.

Files:

* `navigation/grid_path.py` (`bfs_path`, `dijkstra_center_path`)

### Polyline simplification + smoothing

* Simplify the pixel path using **Ramer–Douglas–Peucker** (RDP)
* Smooth using moving-average (keeps endpoints)

Files:

* `navigation/polyline.py` (`rdp`)
* `navigation/splines.py` (`smooth_polyline`)

### Spline model: centripetal Catmull–Rom + arc-length table

We fit a **centripetal Catmull–Rom spline** through the smoothed control points, then precompute an arc-length table to map between:

* spline parameter $u$ and
* physical distance along curve $s$.

This allows:

* “lookahead by pixels” (distance-based target selection)
* consistent speed along the curve

Files:

* `navigation/splines.py` (`CatmullRom2D`, `build_arclength_table`, `u_from_s`, `eval_s`)

---

# Problem 1 — Single robot follows spline inside corridor

### Goal

Move a single robot from A to B while keeping the robot disk inside the corridor.

### Key ideas

1. **Safe region**: erode corridor by robot radius so that if the robot center stays inside, the whole disk stays inside.

   * `safe_mask255 = erode(mask255, disk(robot_radius))`

2. **Spline target tracking**: robot follows a moving target point on the spline using a **lookahead distance**:

   * find closest progress $s_{\text{closest}}$ (approx using sampled points + local search)
   * maintain a monotonic reference progress $s_{\text{ref}}$:
     $$
     s_{\text{ref}} \leftarrow \max\left(s_{\text{ref}} + v_s\Delta t,, s_{\text{closest}}\right)
     $$
   * set target:
     $$
     T = \text{spline}(s_{\text{ref}} + s_{\text{lookahead}})
     $$

3. **Wall force** (distance field gradient) + a hard safety-net snap if needed.

Files:

* `navigation/problem1.py` (`build_center_spline_from_mask`, `simulate_problem1_single_robot`)
* `navigation/corridor.py` (erosion, distance transform, wall force, snapping)

## Example
![p1](runs/problem1_follow.png)

---

# Problem 2 — Bidirectional swarm inside corridor (A→B and B→A)

### Goal

Two swarms start simultaneously: one at A moving to B, the other at B moving to A, without collisions and without leaving the corridor.

### Key ideas

1. **Two lanes** from the same spline centerline
   For progress $s$, compute tangent $\hat{t}$ and normal $\hat{n}=[-t_y, t_x]$.
   Lane target is:
   $$
   p_{\text{lane}}(s)=p_{\text{center}}(s)+\sigma,d_{\text{lane}}\hat{n}
   $$
   where $\sigma=+1$ for A→B group and $\sigma=-1$ for B→A group.
   If the offset goes outside the safe mask, the offset is **shrunk** until inside.

2. **Progress tracking in opposite directions**

   * forward group increases $s$
   * backward group decreases $s$

3. **Collision handling (two layers)**

   * **Continuous repulsion force** in the IVP: inverse-cubic repulsion if within $R_{\text{safe}}$
   * **Hard constraint projection** after the IVP step (PBD-style): iteratively adjust positions so $\lVert x_i-x_j\rVert \ge d_{\min}$

   Projection rule for a violating pair $(i,j)$:
   $$
   \delta = \frac{1}{2}(d_{\min}-d),\hat{r}, \quad \hat{r}=\frac{x_i-x_j}{\lVert x_i-x_j\rVert}
   $$
   then:
   $$
   x_i \leftarrow x_i + \delta,\quad x_j \leftarrow x_j - \delta
   $$

4. After projection, velocities are recomputed from corrected positions:
   $$
   v_{n+1}=\frac{x_{n+1}-x_n}{\Delta t}
   $$
   then saturated.

Files:

* `navigation/problem2.py` (`simulate_problem2_bidirectional`, lane logic, spacing, collision projection)
* `navigation/dynamics.py` (IVP step + repulsion)
* `navigation/corridor.py` (safe mask + wall force)

---

# Problem 3 — Robot navigation in pedestrian flow (two directions)

This repo supports two workflows:

## (A) Synthetic pedestrian-flow video + ground truth (recommended for reproducibility)

`navigation/p3_synth_video.py` generates:

* a synthetic corridor background
* pedestrians moving in both directions on “lanes”
* an `.npz` file containing:

  * `ped_pos` with shape $(T,K,2)$
  * corridor masks and safe masks
  * A and B points
  * radii and fps

The robot simulation (`navigation/p3_sim.py`) then:

1. builds clearance + gradient on the robot-safe mask
2. tracks a spline lookahead target (A→B or B→A) via `dir_sign = ±1`
3. applies:

   * pedestrian repulsion (inverse-cubic inside a safety radius)
   * wall force from clearance gradient
4. enforces corridor safety:

   * if a proposed step leaves the safe mask, it “bounces” by reflecting velocity across inward normal (derived from $\nabla d$):
     $$
     v' = v - 2(v\cdot \hat{n})\hat{n}
     $$

Files:

* `navigation/p3_synth_video.py` (video + GT generator)
* `navigation/p3_io.py` (load GT `.npz`, overlay helpers)
* `navigation/p3_sim.py` (robot simulation in ped flow)

## (B) Real video pedestrian flow (background subtraction)

`scripts/p3_run_real_video.py`:

1. estimates a median background image
2. extracts corridor mask from background
3. user clicks A and B on the path (OpenCV UI)
4. detects pedestrians per frame using background subtraction + contour centroids
5. simulates a robot that does a **round trip** (A→B then B→A)

Pedestrian repulsion in this real-video script is a **linear radial push** (simpler and more stable with noisy detections):
$$
f_{\text{ped}} += k_{\text{ped}}(R_{\text{safe}}-d),\hat{n}\quad \text{for } d<R_{\text{safe}}
$$

---

## Repository structure

* `navigation/`

  * `preprocess.py` — corridor mask extraction from image
  * `grid_path.py` — BFS / Dijkstra centerline path
  * `polyline.py` — RDP polyline simplification
  * `splines.py` — Catmull–Rom spline + arc-length table
  * `dynamics.py` — IVP step, saturation, repulsion forces
  * `corridor.py` — safe-mask erosion, distance transform + gradient, wall force, snapping
  * `problem1.py` — single robot along spline inside corridor
  * `problem2.py` — bidirectional swarm (two lanes + collision handling)
  * `p3_synth_video.py` — synthetic pedestrian corridor + GT writer
  * `p3_io.py` — load `.npz` GT + overlay helpers
  * `p3_sim.py` — simulate robot in pedestrian flow (two directions)
  * `replay_render.py` — render replay `.npz` (P1/P2) into a video
  * `ui.py` — click-based A/B selection (OpenCV)

* `scripts/`

  * `demo_core.py` — tiny IVP demo (robot chasing target)
  * `find_path_ab.py` — preprocess + compute A→B path on an image (synthetic or real)
  * `export_replay_video.py` — convert saved replay `.npz` into mp4/avi
  * `p3_make_synth_video.py` — generate synthetic pedestrian video + GT
  * `p3_run_synth.py` — run P3 simulation on synthetic GT and render overlay
  * `p3_run_real_video.py` — run P3 on a real pedestrian video

---

## Installation

Python packages used:

* `numpy`
* `opencv-python`
* `matplotlib`

Typical install:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install numpy opencv-python matplotlib
```

---

## How to run

### 0) Core IVP demo

```bash
python -m scripts.demo_core --steps 300 --dt 0.05 --out runs/demo_core.png
```

### 1) Extract an A→B path on a map (mask + centerline path)

Synthetic example:

```bash
python -m scripts.find_path_ab --synthetic --method center --out runs/ab_path.png
```

Real image example:

```bash
python -m scripts.find_path_ab --in path/to/map.png --invert auto --method center --out runs/ab_path.png
```

---

## Problem 3 runs

### A) Make synthetic pedestrian video + GT

```bash
python -m scripts.p3_make_synth_video ^
  --out_video runs/p3_synth.mp4 ^
  --out_gt runs/p3_synth_gt.npz ^
  --frames 1200 --fps 30 ^
  --n_each_dir 20 ^
  --ped_r 14 --robot_r 10 ^
  --corridor_width 280 --lane 90 ^
  --speed 90 --jitter 1.5 --seed 42
```

### B) Run robot simulation on synthetic GT (two directions)

```bash
python -m scripts.p3_run_synth --synthetic --out_dir runs --corridor_width 280 ^
  --k_p 16 --k_d 8 --k_ped 1400 --v_max 140 ^
  --lookahead 40 --wall_margin 6 --k_wall 120
```

### C) Run on a real pedestrian video (background subtraction + clicks)

```bash
python -m scripts.p3_run_real_video --video path/to/pedestrians.mp4 --out runs/p3_real_output.mp4 --invert auto
```

Controls worth tuning on real videos:

* `--invert` (fixes corridor mask polarity if thresholding is flipped)
* `--k_ped` (how strongly the robot avoids pedestrians)
* `--k_wall` (how strongly it avoids corridor walls)
* `--v_max` (caps speed; too high can cause jitter/bouncing)

---

## Replay export (Problem 1 / Problem 2)

`navigation/replay_render.py` can render a saved replay `.npz` into a video.
Expected `.npz` fields:

* `bg_bgr` : background image (H,W,3) uint8
* `mask255` : corridor mask (H,W) uint8 (optional but recommended)
* `safe_mask255` : eroded safe mask (H,W) uint8 (optional but recommended)
* `traj` : positions

  * P1: (T,2)
  * P2: (T,N,2)
* `robot_radius_px` : int
* optional `fps`
* optional `group_split` (for P2 coloring: first group `[0:split]`, second `[split:N]`)

Export:

```bash
python -m scripts.export_replay_video --npz runs/p2_replay.npz --out runs/p2_replay.mp4
```

Disable trails:

```bash
python -m scripts.export_replay_video --npz runs/p2_replay.npz --out runs/p2_replay.mp4 --no_trail
```

---

## Notes on design choices

* **Eroding the corridor mask** is the cleanest way to enforce “robot disk never crosses walls” using only the robot center.
* **Distance transform gradients** give a smooth inward direction, so wall avoidance behaves like a soft potential field rather than a brittle hard constraint.
* **Dijkstra with clearance penalty** produces a centerline-like path even for wide corridors (it naturally avoids hugging boundaries).
* **Semi-implicit Euler** is used because it is robust for damped spring-like systems.
* **Problem 2** uses both:

  * continuous repulsion (prevents many collisions), and
  * hard projection (guarantees minimum separation even in tight encounters).
* **Problem 3 (real video)** uses simpler pedestrian forces (linear radial) because detections are noisy; inverse-cubic can be too sensitive to jitter.

---

## Quick "what to show" in the report/demo

* Problem 1: corridor mask → centerline spline → robot trajectory with safe region tint
* Problem 2: two swarms passing each other (different colors per direction) + minimum distance over time
* Problem 3: robot overlaid on pedestrians (synthetic GT or real video), demonstrating at least two directions
