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