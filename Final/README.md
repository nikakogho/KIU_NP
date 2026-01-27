NP final project nikak

## Modeling choice per sub-problem

We simulate a synchronized swarm of (N) drones in 3D using **initial value problems (IVPs)**: we integrate ordinary differential equations forward in time from given initial positions/velocities until the swarm converges to the desired configuration. Each drone state is
$(x_i, v_i) \in \mathbb{R}^3 \times \mathbb{R}^3$. We enforce:

1.  **Velocity saturation:** $\|v_i\| \le v_{\max}$
2.  **Collision avoidance:** Via a repulsive force activated within a safety radius $R_{\text{safe}}$
3.  **Damping:** To prevent oscillations.

### Sub-problem 1: Static formation (handwritten name)
**Chosen model:** IVP with **position target tracking** (Method 1).  
We convert the handwritten-name image into a set of (N) 3D anchor points T<sub>i</sub> (typically on a plane so z=0). Each drone is assigned to one anchor point and follows dynamics that attract it to T<sub>i</sub> while repelling nearby drones and saturating speed.  

**Why it fits:** this is a “go from initial state to a fixed final shape” task; IVP integration naturally yields smooth, collision-aware trajectories.

### Sub-problem 2: Transition to “Happy New Year!”
**Chosen model:** IVP with **time-varying position targets** (Method 1 with T<sub>i</sub>(t)).  
We generate a second anchor set for the greeting and define a smooth morph T<sub>i</sub>(t) from the name anchors to the greeting anchors over a transition interval. The same controller as (1) produces continuous trajectories.

**Why it fits:** reuses the same stable formation controller; smooth target interpolation prevents abrupt motion and reduces collisions.

### Sub-problem 3: Dynamic tracking + shape preservation (video)
**Chosen model:** IVP with **velocity tracking** (Method 3 / IVP-VT) **plus an explicit formation preservation term**.  
A moving object is tracked from the video to produce a motion signal (e.g., object position/velocity over time or a local velocity field V(x,t)). Drones match a saturated reference velocity V<sub>sat</sub> while maintaining a formation (fixed offsets to a moving frame, or neighbor-distance constraints) and avoiding collisions.

**Why it fits:** video motion is naturally expressed as velocity; velocity tracking yields responsive following. A formation term is necessary because repulsion alone prevents collisions but does not guarantee shape preservation.

### Why we do not use BVPs
We avoid boundary value problem (BVP) solvers because:

- **Inequality constraints** like collision avoidance ($\|x_i-x_j\| \ge R_{\text{safe}}$) and speed limits ($\|v_i\| \le v_{\max}$) are awkward to enforce in standard BVP formulations.
- **Numerical brittleness:** BVPs are typically more brittle numerically for large coupled systems ($6N$ states) and require careful initial guesses.
- **Natural fit for IVP:** The project inputs are naturally "start here and move forward," making **IVP simulation** simpler to implement, easier to validate, and easier to debug while still producing smooth trajectories.

## Anchoring

This project converts each desired figure (handwritten name, “Happy New Year!”) into **exactly N 3D anchor points**
(one per drone). Drones are then assigned to anchors and track them with collision avoidance.

We first render / load the figure as a **binary mask** `M` on a fixed canvas (e.g. 1200×300):
- `M[y,x] = 1` means “this pixel belongs to the figure”
- `M[y,x] = 0` means background

From `M` we derive three candidate point sets:

### Candidate sets (B, K, F)

1) **Boundary / Outline (B)**  
Boundary pixels form the silhouette, which is the most important cue for legibility in sparse dot displays.  
Computed as:
- `B = M AND (NOT erosion(M))`

2) **Skeleton / Stroke centerline (K)**  
A topology-preserving, 1-pixel-wide centerline of the figure, useful when N is small (handwriting becomes readable).  
Computed via morphological skeletonization (iterated erosion/opening).

3) **Fill / Interior (F)**  
All remaining foreground pixels (optionally excluding the boundary) to make letters look “solid” when N is large.  
Computed as:
- `F = M AND (NOT B)`  (or simply all foreground pixels if desired)

### Adaptive allocation of drones across B/K/F

We adapt the visual style based on how many drones we have relative to the figure’s complexity.

Let:
- `P = |B|` be the boundary pixel count
- `L = |K|` be the skeleton pixel count
- `N` be the number of drones

Define a density score:

$$
d = \frac{N}{L + 0.35P}
$$

We use three regimes:

- **Sparse** (`d < 0.6`): maximize legibility with centerlines  
  - `nK = N`, `nB = 0`, `nF = 0`

- **Medium** (`0.6 ≤ d < 1.5`): add silhouette while keeping strokes readable  
  - `nB = floor(0.35N)`, `nK = N - nB`, `nF = 0`

- **Dense** (`d ≥ 1.5`): crisp outline + solid interior  
  - `nB = floor(0.25N)`, `nF = N - nB`, `nK = 0` (optional: keep a small `nK` for handwriting)

### Evenly-spaced sampling (anti-clumping)

From each candidate set (B, K, F) we must pick exactly `nB`, `nK`, `nF` points.
We do **evenly-spaced sampling** (greedy farthest-point / Poisson-disk-like) rather than uniform random sampling to avoid clumps and improve readability.

Sampling order is:
1) Boundary points (if used)
2) Skeleton points (if used)
3) Fill points (if used)

Finally, selected pixel coordinates are mapped into 3D world coordinates on a display plane (typically `z = 0`),
scaled and centered consistently across all sub-problems.
