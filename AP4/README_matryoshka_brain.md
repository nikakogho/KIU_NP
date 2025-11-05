# Matrioshka Brain–Style Dyson Swarm: Multi-Shell Spherical Voronoi (approx) + k-NN Links

## What this project does

This version generalizes the single-shell Dyson Swarm to a Matrioshka configuration with multiple concentric spherical shells at different radii. For each shell it:

1. places modules quasi-uniformly over the sphere,
2. builds an approximate Delaunay-like neighbor graph using k-nearest neighbors by angular distance,
3. rasterizes a spherical Voronoi partition (nearest-module by angle), and
4. renders:
   • a 3D scene with all shells, intra-shell great-circle links, and optional inter-shell radial links
   • one equirectangular Voronoi map per shell.

All figures are saved to the plots folder (auto-created if missing).

## Key ideas and approach

* Concentric shells: Sites are defined on multiple spheres with chosen radii. Geometry is computed in unit directions (for correct angular logic) and then scaled by radius for plotting.
* Per-shell angular Voronoi: Each shell gets its own spherical Voronoi partition by angular nearest neighbor on that shell; shells do not compete with each other for territory.
* Delaunay-like graph via k-NN: True spherical Delaunay triangulation is complex; a k-NN graph on unit directions approximates local connectivity and is easy to compute.
* Great-circle arcs: Intra-shell links are drawn along great circles on the unit sphere and then scaled to that shell’s radius for 3D rendering.
* Optional radial links: For visualization across shells, each site on shell s can connect to its angularly nearest site on shell s±1 with straight line segments.

## Modules and helpers

* Normalization: Projects any perturbed 3D vector back to the unit sphere.
* Spherical/Cartesian transforms:

  * sph2cart converts latitude/longitude to 3D points on a sphere of radius r.
  * cart2sph converts 3D points to latitude/longitude.
* Great-circle interpolation: Uses slerp to sample points along arcs between two unit directions for smooth, geodesic links.
* Fibonacci sphere sampler: Generates quasi-uniform unit directions; optional small Cartesian jitter followed by renormalization breaks perfect regularity.

## Pipeline

1. Generate per-shell sites

   * For each requested radius, sample N module directions on the unit sphere and scale them by that radius for plotting.
   * Keep both: unit directions for angular math, scaled positions for 3D visuals.

2. Build per-shell neighbor graphs

   * Compute dot-product similarity matrix among unit directions.
   * For each site, select k highest-dot neighbors (largest cosine, smallest angular distance).
   * Symmetrize to obtain undirected edges.

3. Optional inter-shell (radial) links

   * For each adjacent shell pair, connect each site on the inner shell to its angularly nearest neighbor on the next shell.

4. Per-shell Voronoi rasterization

   * Create a longitude-latitude grid on the unit sphere.
   * For each grid point, assign the site index with maximum dot product (nearest by angle) among that shell’s sites.
   * Process in chunks to limit memory footprint.

5. Plotting and export

   * 3D: all wireframe spheres (one per radius), per-shell site markers, great-circle arcs for intra-shell k-NN links, and optional straight radial links.
   * 2D: one equirectangular Voronoi map per shell with overlaid site long/lat points.
   * Save all figures in plots.

## Parameters to tune

* radii: list of shell radii for the Matrioshka configuration.
* N_modules_per_shell: list of module counts for each corresponding shell.
* jitter and seed: control sampling randomness and reproducibility.
* k_nn: number of neighbors in the per-shell k-NN graph; controls mesh sparsity.
* nlon, nlat: raster resolution for Voronoi maps (longitude × latitude).
* chunk: batch size for raster evaluation; trade-off between speed and memory.
* ADD_RADIAL_LINKS and RADIAL_LINKS_PER_SITE: whether to draw inter-shell links and how many nearest neighbors to use between adjacent shells.

## Outputs

* plots/dyson_matrioshka_knn.png
  A 3D visualization showing:

  * Concentric wireframe spheres at the requested radii.
  * Site markers distributed on each shell.
  * Great-circle arcs forming the per-shell k-NN graphs.
  * Optional straight line segments linking angularly nearest sites across adjacent shells.
    What to look for: relative shell sizes, uniformity of site placement, local connectivity within shells, and cross-shell structure if radial links are enabled.

* plots/voronoi_shell_r<radius>.png (one per shell)
  Equirectangular maps (longitude on the horizontal axis, latitude on the vertical) showing:

  * Color-coded spherical Voronoi regions for that shell’s sites (nearest by angle).
  * Overlaid site positions in long/lat.
    What to look for: near-uniform coverage with cell shapes that vary by sampling and jitter; projection-induced distortions near the poles are artifacts of the map, not of the underlying spherical geometry.

## Notes on accuracy and performance

* Angular Voronoi approximation: The Voronoi boundaries are rasterized; increasing nlon and nlat improves outline crispness at higher compute cost.
* k-NN vs true Delaunay: The k-NN graph is a practical surrogate for local adjacency; larger k increases density and visual clutter while deviating from minimal triangulations.
* Chunked rasterization: Controls peak memory for large grids and many shells.

## Practical usage

* Adjust radii and N_modules_per_shell to explore different Matrioshka layouts, densities, and layering strategies.
* Tune k_nn to balance connectivity detail and readability.
* Increase grid resolution for publication-quality maps; adjust chunk to fit your machine’s memory.

## Conceptual takeaways

* Separating direction (unit sphere) from scale (radius) keeps spherical geometry correct and makes multi-shell rendering straightforward.
* Angular proximity (via dot products) is the natural metric for on-sphere nearest-neighbor tasks; it is fast and numerically stable.
* k-NN graphs on the sphere provide a simple, scalable stand-in for more complex spherical triangulations when the goal is qualitative structure and visualization across multiple shells.
