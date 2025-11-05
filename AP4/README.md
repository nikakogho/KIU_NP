# Spherical Voronoi & “Delaunay-like” Dyson Swarm Shell

## What this project does

This script generates a quasi-uniform set of satellite “modules” on a unit sphere (a stylized Dyson Swarm shell), builds an approximate neighbor graph on the sphere, rasterizes a spherical Voronoi partition (nearest-module regions), and produces two figures:

1. a 3D sphere with sites and great-circle links, and
2. an equirectangular map coloring each pixel by its nearest site.

All outputs are saved into the plots folder (auto-created if missing).

## Key ideas and approach

* Sphere representation: Points on the sphere are 3D unit vectors (x, y, z) with norm 1 (could expand for more ambitious takes of "layered dyson sphere"/"matryoshka brain" with multiple surfaces around the sun but keeping it simple for this task).
* Site distribution: Uses a Fibonacci spiral (a standard low-discrepancy method) to place N modules nearly uniformly on the sphere; optional small random jitter is added and then re-normalized to the sphere.
* Distance on a sphere: Angular distance is computed via the arccosine of the dot product between unit vectors; for nearest-neighbor tasks we work directly with dot products to avoid expensive arccos everywhere.
* Delaunay-like neighbor graph: True spherical Delaunay triangulation is nontrivial. We approximate it by connecting each site to its k nearest neighbors on the sphere (based on largest dot products), then draw great-circle arcs between them.
* Spherical Voronoi via raster: Instead of constructing exact spherical Voronoi cells, we rasterize the sphere in longitude/latitude and label each pixel by the nearest site (chunked to control memory).

## Modules and helpers

* Normalization: Ensures any perturbed 3D vector is projected back to the unit sphere.
* Coordinate transforms:
  * sph2cart converts latitude/longitude (radians) to a 3D unit vector.
  * cart2sph converts a 3D unit vector to latitude/longitude (radians).
* Great-circle arc sampling: Uses spherical linear interpolation (slerp) to draw smooth arcs between neighbor sites on the sphere for visualization.

## Pipeline

1. Generate Dyson sites
   * N_modules points via Fibonacci sphere with a small jitter to break perfect regularity.
   * All points are re-normalized to lie exactly on the unit sphere.
2. Build neighbor graph
   * Compute dot-product matrix between all sites.
   * For each site, choose k highest-dot neighbors.
   * Symmetrize to get undirected edges.
   * These edges are drawn as great-circle arcs in the 3D figure.
3. Rasterize spherical Voronoi
   * Create an equirectangular grid in longitude × latitude.
   * Convert each grid cell center to 3D and compute dot products to all sites.
   * Assign the site with the largest dot product (equivalently, smallest angular distance).
   * Done in chunks to avoid large memory spikes.
   * The result is a 2D label image aligned to long/lat axes.
4. Plot and save
   * Figure 1 (3D): wireframe sphere, site points, and great-circle arcs for the k-NN graph.
   * Figure 2 (map): colored label image (default colormap) with overlaid site long/lat positions.

## Parameters to tune
* N_modules: number of Dyson swarm modules on the shell.
* jitter: magnitude of the small Gaussian perturbation before re-normalization.
* k: number of nearest neighbors used to approximate Delaunay adjacency.
* nlon, nlat: longitude/latitude raster resolution for the Voronoi map.
* chunk: batch size for raster processing to balance speed and memory.

## Outputs
* plots/dyson_sphere_knn.png
  3D visualization of the unit sphere with:
  * Light wireframe sphere as reference.
  * Sites as small markers distributed over the surface.
  * Great-circle arcs connecting each site to its k nearest neighbors.
    What to look for: The overall uniformity of site placement and the local, roughly triangular mesh implied by the k-NN links.
* plots/dyson_sphere_voronoi_map.png
  Equirectangular map (longitude on the horizontal axis, latitude on the vertical) showing:
  * Each pixel colored by the nearest site (spherical Voronoi region).
  * Overlaid site positions in longitude/latitude.
    What to look for: Voronoi cells of varying shapes and sizes; near-uniform coverage if N_modules is large and jitter is small, with distortions near the poles due to the map projection (not the underlying spherical geometry).

## Notes on accuracy and performance
* The Voronoi shown on the map is a nearest-site raster approximation. Increasing nlon and nlat yields crisper, more accurate boundaries at higher compute cost.
* The “Delaunay-like” graph is a k-NN surrogate; small k produces a sparse mesh, large k increases visual clutter and deviates from Delaunay-like locality.
* Chunked raster evaluation keeps memory bounded when nlon × nlat is large.

## Practical usage
* Edit N_modules, jitter, k, and grid resolution to explore different swarm densities and neighbor structures.
* The plots directory is created automatically; generated figures are saved there and also shown on screen.

## Conceptual takeaways
* On a sphere, dot products between unit vectors are a fast proxy for angular proximity, enabling efficient nearest-neighbor operations.
* k-NN graphs on the sphere can serve as a simple stand-in for more complex spherical triangulations when you mainly need reasonable local connectivity.
* Rasterized nearest-site labeling provides an intuitive, scalable way to visualize spherical Voronoi partitions without implementing exact spherical Voronoi geometry.
