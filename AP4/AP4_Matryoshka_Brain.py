# Matrioshka Brain Style Dyson Swarm: Multi-Shell Spherical Voronoi (approx) + k-NN Links
# - Per-shell "Voronoi" is computed by nearest-neighbor on the sphere (angular distance)
#   over a dense longitude/latitude grid (chunked to avoid memory explosion).
# - Per-shell "Delaunay-like" neighbor graph is approximated with k-NN great-circle arcs.
# - Multiple concentric shells are supported; geometry math uses unit directions for angle,
#   then scales by shell radius for plotting.
#
# Outputs:
#   plots/matryoshka/dyson_matrioshka_knn.png                -> 3D plot with all shells + k-NN links
#   plots/matryoshka/voronoi_shell_r<radius>.png (per shell) -> equirectangular Voronoi map

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
path = "plots/matryoshka"
os.makedirs(path, exist_ok=True)

# Parameters (tweak as desired)
radii = [1.0, 1.3, 1.6]            # shell radii (arbitrary units)
N_modules_per_shell = [250, 400, 600]  # number of modules per shell
jitter = 0.02                      # small Gaussian jitter before re-normalization
seed = 42                          # RNG seed for reproducibility
k_nn = 4                           # k for k-NN neighbor graph (per shell)

# Voronoi raster resolution (per shell)
nlon = 600
nlat = 300
chunk = 6000

# Optional: draw simple "radial" links between adjacent shells (for visual context)
ADD_RADIAL_LINKS = True
RADIAL_LINKS_PER_SITE = 1  # connect each site to its single angularly nearest neighbor on adjacent shells

# Helpers: Sphere Math
def normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def sph2cart(lat, lon, r=1.0):
    """
    lat, lon in radians (lat = [-pi/2, pi/2], lon = [-pi, pi])
    Returns xyz on sphere of radius r.
    """
    cl = np.cos(lat)
    x = r * cl * np.cos(lon)
    y = r * cl * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)

def cart2sph(xyz):
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    lat = np.arctan2(z, np.sqrt(x*x+y*y))
    lon = np.arctan2(y, x)
    return lat, lon

def great_circle_arc(p, q, num=100):
    """
    Return points along the great-circle arc from p to q on the unit sphere,
    using spherical linear interpolation (slerp). p, q are (3,) unit vectors.
    """
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        # nearly identical: return a tiny arc
        return np.vstack([p, q])
    theta = np.arccos(dot)
    t = np.linspace(0, 1, num)
    s1 = np.sin((1-t)*theta) / np.sin(theta)
    s2 = np.sin(t*theta) / np.sin(theta)
    pts = (s1[:,None]*p + s2[:,None]*q)
    return normalize(pts)

# Dyson swarm "sites" per shell
def fibonacci_sphere(n, jitter=0.0, seed=7):
    """
    Quasi-uniform points on the unit sphere (Fibonacci spiral).
    Optionally add small Cartesian jitter then re-normalize.
    Returns unit vectors (n, 3).
    """
    rng = np.random.default_rng(seed)

    ga = np.pi * (3. - np.sqrt(5.))   # golden angle
    i = np.arange(n)
    z = (2.*i + 1.)/n - 1.            # in (-1,1)
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    theta = i * ga
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
        pts = normalize(pts)
    return pts

# Generate shells (unit directions + scaled positions)
assert len(radii) == len(N_modules_per_shell), "radii and N_modules_per_shell must match in length."

all_shells_dirs = []   # list of (Ni,3) unit vectors per shell
all_shells_pts  = []   # list of (Ni,3) scaled positions per shell
for r, Ni in zip(radii, N_modules_per_shell):
    dirs = fibonacci_sphere(Ni, jitter=jitter, seed=seed)
    pts  = dirs * r
    all_shells_dirs.append(dirs)
    all_shells_pts.append(pts)

# Delaunay-like neighbor graph (per shell)
def angular_distance_matrix(u, v):
    # For unit vectors, angular distance = arccos(dot); we only need dot for k-NN
    dots = np.clip(u @ v.T, -1.0, 1.0)
    return np.arccos(dots), dots

def knn_edges_on_sphere_dirs(unit_dirs, k=4):
    """
    k-NN neighbors on the unit sphere using dot-product similarity.
    Returns a sorted list of undirected edges (i, j) with i<j.
    """
    _, dots = angular_distance_matrix(unit_dirs, unit_dirs)
    np.fill_diagonal(dots, -np.inf)
    idx = np.argpartition(-dots, kth=np.arange(k), axis=1)[:, :k]
    edges = set()
    for i in range(unit_dirs.shape[0]):
        for j in idx[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)

shell_edges = [knn_edges_on_sphere_dirs(d, k=k_nn) for d in all_shells_dirs]

# Optional: simple radial links between adjacent shells (nearest by angle)
radial_links = []  # list of lists for each between-shell pair: [(i_in_shell_s, j_in_shell_s+1), ...]
if ADD_RADIAL_LINKS and len(radii) > 1:
    for s in range(len(radii)-1):
        U = all_shells_dirs[s]      # (Ns,3)
        V = all_shells_dirs[s+1]    # (Ns+1,3)
        # For each u in U, find angularly nearest v in V (max dot)
        dots = np.clip(U @ V.T, -1.0, 1.0)
        j_best = np.argmax(dots, axis=1)
        links = [(i, int(j_best[i])) for i in range(U.shape[0])]
        radial_links.append(links)

# Spherical Voronoi (per shell) by raster (equirectangular)
def voronoi_sphere_labels_for_shell(unit_dirs, nlon=720, nlat=360, chunk=8000):
    """
    Compute nearest-site labels for a lon/lat raster for a single shell.
    "Nearest" by angular similarity (max dot) to unit_dirs.
    Returns (labels, lon_grid, lat_grid).
    """
    lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False)
    lats = np.linspace(-np.pi/2, np.pi/2, nlat)  # include poles
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten grid to chunks (unit sphere)
    grid_xyz = sph2cart(lat_grid.ravel(), lon_grid.ravel(), r=1.0)
    labels = np.empty(grid_xyz.shape[0], dtype=np.int32)

    S = unit_dirs  # (M,3), unit
    for start in range(0, grid_xyz.shape[0], chunk):
        end = min(start + chunk, grid_xyz.shape[0])
        G = grid_xyz[start:end]                      # (g,3)
        dots = np.clip(G @ S.T, -1.0, 1.0)          # (g, M)
        labels[start:end] = np.argmax(dots, axis=1) # nearest site index
    labels = labels.reshape(lat_grid.shape)
    return labels, lon_grid, lat_grid

shell_labels = []
for d in all_shells_dirs:
    L, LG, LT = voronoi_sphere_labels_for_shell(d, nlon=nlon, nlat=nlat, chunk=chunk)
    shell_labels.append((L, LG, LT))

# 3D Plot: all shells + k-NN arcs (+ optional radial links)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# wireframe spheres at each radius
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
for r in radii:
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, rstride=2, cstride=2)

# sites and intra-shell arcs
for s, (pts_r, dirs, edges) in enumerate(zip(all_shells_pts, all_shells_dirs, shell_edges)):
    # sites
    ax.plot(pts_r[:,0], pts_r[:,1], pts_r[:,2], marker='o', linestyle='')
    # arcs along great circles on that shell
    for i, j in edges:
        arc_unit = great_circle_arc(dirs[i], dirs[j], num=40)
        arc = arc_unit * radii[s]
        ax.plot(arc[:,0], arc[:,1], arc[:,2], linewidth=0.6)

# optional: simple radial links between adjacent shells
if ADD_RADIAL_LINKS and len(radii) > 1:
    for s, links in enumerate(radial_links):
        r0, r1 = radii[s], radii[s+1]
        dirs0 = all_shells_dirs[s]
        dirs1 = all_shells_dirs[s+1]
        for i0, j1 in links:
            p0 = dirs0[i0] * r0
            p1 = dirs1[j1] * r1
            # draw a straight segment between shells (not a great circle)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=0.4)

ax.set_xlim(-radii[-1]*1.1, radii[-1]*1.1)
ax.set_ylim(-radii[-1]*1.1, radii[-1]*1.1)
ax.set_zlim(-radii[-1]*1.1, radii[-1]*1.1)
ax.set_title("Matrioshka Dyson Swarm: Multi-Shell Sites + k-NN Great-Circle Links")
plt.savefig(os.path.join(path, "dyson_matrioshka_knn.png"), bbox_inches='tight', dpi=160)
plt.show()

# 2D Voronoi maps per shell
for s, (L, LG, LT) in enumerate(shell_labels):
    plt.figure()
    plt.imshow(L, origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
    # overlay site positions (convert unit dirs to lat/lon)
    lat_s, lon_s = cart2sph(all_shells_dirs[s])
    plt.plot(np.degrees(lon_s), np.degrees(lat_s), marker='o', linestyle='')
    plt.title(f"Spherical Voronoi (angular) — Shell r={radii[s]:.2f}")
    plt.xlabel("Longitude (deg)"); plt.ylabel("Latitude (deg)")
    fname = os.path.join(path, f"voronoi_shell_r{radii[s]:.2f}.png")
    plt.savefig(fname, bbox_inches='tight', dpi=160)
    plt.show()
