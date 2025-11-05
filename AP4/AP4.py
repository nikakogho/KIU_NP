# Spherical Voronoi/Delaunay (approx) for a Dyson Swarm shell
# - "Voronoi" is computed by nearest-neighbor on the sphere (angular distance)
#   over a dense longitude/latitude grid (chunked to avoid memory explosion).
# - "Delaunay-like" neighbor graph is approximated with k-NN great-circle arcs.

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
path = "plots"
os.makedirs(path, exist_ok=True)

# Helpers: sphere math

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def sph2cart(lat, lon):
    """
    lat, lon in radians (lat = [-pi/2, pi/2], lon = [-pi, pi])
    Returns xyz on unit sphere.
    """
    cl = np.cos(lat)
    x = cl * np.cos(lon)
    y = cl * np.sin(lon)
    z = np.sin(lat)
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

# Dyson swarm "sites"

def fibonacci_sphere(n, jitter=0.0, seed=7):
    """
    Quasi-uniform points on the unit sphere (Fibonacci spiral).
    Optionally add small Cartesian jitter then re-normalize.
    """
    rng = np.random.default_rng(seed)

    # Golden angle
    ga = np.pi * (3. - np.sqrt(5.))
    i = np.arange(n)
    z = (2.*i + 1.)/n - 1.   # in (-1,1)
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    theta = i * ga
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
        pts = normalize(pts)
    return pts

# Choose number of modules on shell
N_modules = 250
sites = fibonacci_sphere(N_modules, jitter=0.02, seed=42)   # Dyson shell snapshot

# Delaunay-like neighbor graph (k-NN on sphere)

def angular_distance_matrix(u, v):
    # For unit vectors, angular distance = arccos(dot); we only need dot for k-NN
    dots = np.clip(u @ v.T, -1.0, 1.0)
    return np.arccos(dots), dots

def knn_edges_on_sphere(sites, k=4):
    # Use dot-products to find k nearest neighbors (largest dots)
    _, dots = angular_distance_matrix(sites, sites)
    np.fill_diagonal(dots, -np.inf)  # ignore self
    idx = np.argpartition(-dots, kth=np.arange(k), axis=1)[:, :k]
    # Make undirected edges
    edges = set()
    for i in range(sites.shape[0]):
        for j in idx[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)

edges = knn_edges_on_sphere(sites, k=4)

# Spherical Voronoi by raster projection (equirectangular)

def voronoi_sphere_labels(sites, nlon=720, nlat=360, chunk=8000):
    """
    Compute nearest-site labels for a lon/lat raster. Chunked to limit memory.
    Returns (labels, lon_grid, lat_grid).
    """
    lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False)
    lats = np.linspace(-np.pi/2, np.pi/2, nlat)  # include poles
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Flatten grid to chunks
    grid_xyz = sph2cart(lat_grid.ravel(), lon_grid.ravel())
    labels = np.empty(grid_xyz.shape[0], dtype=np.int32)
    # Precompute site vectors
    S = sites  # (M,3), already unit
    M = S.shape[0]
    # Process in chunks
    for start in range(0, grid_xyz.shape[0], chunk):
        end = min(start + chunk, grid_xyz.shape[0])
        G = grid_xyz[start:end]                       # (g,3)
        # Use dot products to measure angular similarity (max dot = nearest)
        dots = np.clip(G @ S.T, -1.0, 1.0)           # (g, M)
        labels[start:end] = np.argmax(dots, axis=1)  # nearest site index
    labels = labels.reshape(lat_grid.shape)
    return labels, lon_grid, lat_grid

labels, lon_grid, lat_grid = voronoi_sphere_labels(sites, nlon=600, nlat=300, chunk=6000)

# Plots

# Figure 1: 3D sphere with sites and k-NN great-circle arcs
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Sphere wireframe
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z, linewidth=0.3, rstride=2, cstride=2)

# Sites
ax.plot(sites[:,0], sites[:,1], sites[:,2], marker='o', linestyle='')

# Great-circle arcs for edges
for i, j in edges:
    arc_pts = great_circle_arc(sites[i], sites[j], num=40)
    ax.plot(arc_pts[:,0], arc_pts[:,1], arc_pts[:,2], linewidth=0.6)

ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
ax.set_title("Dyson Swarm Shell: 3D Sites + k-NN Great-Circle Links")
plt.savefig("plots/dyson_sphere_knn.png", bbox_inches='tight', dpi=160)
plt.show()

# Figure 2: Equirectangular Voronoi map (labels as image)
# Note: we do not set a colormap explicitly; Matplotlib default will be used.
plt.figure()
plt.imshow(labels, origin='lower', extent=[-180, 180, -90, 90], aspect='auto')
# Overlay site positions
site_lat, site_lon = cart2sph(sites)
plt.plot(np.degrees(site_lon), np.degrees(site_lat), marker='o', linestyle='')
plt.title("Spherical Voronoi (nearest-module) — Equirectangular Projection")
plt.xlabel("Longitude (deg)"); plt.ylabel("Latitude (deg)")
plt.savefig("plots/dyson_sphere_voronoi_map.png", bbox_inches='tight', dpi=160)
plt.show()
