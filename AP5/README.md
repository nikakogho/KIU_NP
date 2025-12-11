# Kanji Splines – Comparing Cubic vs Smoothing Splines on Handwritten Strokes

This repository contains a small experimental pipeline that takes raster images of kanji characters and fits **cubic interpolating splines** and **smoothing splines** to their strokes. The goal is to see how the choice of spline affects:

* **Accuracy** (how close the spline stays to the original stroke)
* **Smoothness** (how “wiggly” the spline is)

The experiment is run on three kanji:

* 龍 — *dragon*
* 家 — *house*
* 男 — *man*

---

## Repository Layout

* `kanji_splines.py` – main script with the full pipeline (loading, preprocessing, spline fitting, visualization, metrics).
* `letters/` – source images (PNG) for the kanji.
* `README.md` – this document.

You can treat the script both as a “paper in code” and as a template for other curve-fitting experiments.

---

## High-Level Pipeline

Conceptually, the script does this:

1. **Load & binarize the image**

   * Convert the kanji image to grayscale (`[0,1]`).
   * Threshold to a boolean mask: `True` = ink, `False` = background.

2. **Extract a centerline per column**

   * For each column `x`, find all ink pixels.
   * Take the **mean row index** of those pixels as the “center” of the stroke in that column.
   * This yields a dense centerline `(x, y_center)` sampled at (almost) every integer `x`.

3. **Split into horizontal bands**

   * Divide the image into a fixed number of horizontal bands (e.g. 4).
   * Each band is processed independently, so splines capture local stroke geometry instead of having to model the entire character at once.

4. **Control point selection**

   * For each band:

     * Sort points by `x` and remove duplicate `x` values.
     * Optionally **downsample** to at most `max_points_per_band` control points (e.g. 25).
   * This step is critical: if we kept every single centerline sample, both cubic and smoothing splines would look almost identical; with fewer control points, smoothing has room to differ.

5. **Spline fitting**

   * **Cubic interpolating spline** (`CubicSpline`):

     * Passes exactly through all control points.
   * **Smoothing spline** (`UnivariateSpline` with smoothing factor `s`):

     * Trades off fitting the control points vs. keeping the curve smooth.
     * For large `s`, the spline ignores small irregularities and noise.

6. **Metrics & visualization**

   * For each band:

     * Evaluate both splines at all original dense centerline `x`’s.
     * Compute **MSE** between spline `y` and dense centerline `y`.
     * Compute mean squared **second derivative** as a proxy for curvature/smoothness.
   * Plot:

     * Original kanji
     * Cubic splines overlaid
     * Smoothing splines overlaid

---

## Usage

### 1. Install dependencies

```bash
pip install numpy pillow matplotlib scipy
```

### 2. Prepare images

Place your images in `letters/`, e.g.:

* `letters/dragon kanji.png`
* `letters/house kanji.png`
* `letters/man kanji.png`

Any black-on-white PNG should work.

### 3. Run the experiment

Inside `kanji_splines.py`, configure the image and parameters in the `__main__` block:

```python
if __name__ == "__main__":
    IMAGE_PATH = "letters/dragon kanji.png"
    process_kanji_image(
        image_path=IMAGE_PATH,
        sections=4,        # number of horizontal bands
        threshold=0.5,     # binarization threshold
        smoothing_factor=200.0,
        max_points_per_band=25
    )
```

Then execute:

```bash
python kanji_splines.py
```

You’ll see:

1. Grayscale + binary image.
2. Centerline points per band.
3. Side-by-side comparison of cubic vs smoothing splines.
4. A metrics table printed to the console.

Repeat with `IMAGE_PATH` set to the other kanji images.

---

## Metrics and Findings

For each character we report averages across bands:

* **MSE cubic** – mean squared error between cubic spline and dense centerline.
* **MSE smooth** – mean squared error between smoothing spline and dense centerline.
* **Curv² cubic** – mean squared second derivative of cubic spline.
* **Curv² smooth** – mean squared second derivative of smoothing spline.

### 男 – *man*

| Model     | Avg MSE | Avg Curv² |
| --------- | ------: | --------: |
| Cubic     |  6.6623 |    0.0390 |
| Smoothing | 11.7452 |    0.0113 |

* Interpolating cubic splines are noticeably **more accurate**.
* Smoothing splines are about **3.5× smoother** on average.

---

### 家 – *house*

| Model     | Avg MSE | Avg Curv² |
| --------- | ------: | --------: |
| Cubic     | 11.6125 |    0.0148 |
| Smoothing | 16.8165 |    0.0104 |

* Same overall pattern: cubic wins on MSE, smoothing wins on smoothness.
* One band (with complex geometry) shows that smoothing can sometimes slightly increase curvature locally, reflecting the global nature of the smoothness penalty.

---

### 龍 – *dragon*

| Model     | Avg MSE | Avg Curv² |
| --------- | ------: | --------: |
| Cubic     |  9.3636 |    0.0188 |
| Smoothing | 14.9282 |    0.0020 |

* Here, the smoothing spline is **extremely smooth**: curvature drops by almost an order of magnitude.
* In one band, smoothing even slightly **improves** MSE compared to cubic, acting like a denoiser for noisy centerline samples.

---

## Interpretation

Across all three kanji:

* **Cubic interpolating splines**
  * Reproduce the original centerline very closely.
  * Sensitive to noise: small local irregularities produce visible wiggles.
  * Best when exact reproduction of the observed stroke is critical.
* **Smoothing splines**
  * Sacrifice some accuracy (higher MSE) to produce much smoother curves.
  * Great when you care about stable, physically plausible trajectories more than pixel-perfect fidelity.
  * The effect is strongest when:
    * The smoothing factor `s` is large.
    * The number of control points per band is limited.

In other words, cubic splines are “**what the pixels say**,” smoothing splines are “**what a clean stroke should look like**.”

---

## Parameters You Can Explore

* `sections` – number of horizontal bands
  More bands → more local curves, less global constraint.

* `threshold` – binarization threshold
  Adjust if images are darker/lighter or anti-aliased.

* `max_points_per_band` – number of control points
  Fewer points → stronger interpolation vs smoothing contrast.

* `smoothing_factor` (`s`) – spline smoothness

  * `0` → nearly identical to cubic interpolation.
  * Larger → smoother, but less faithful.

Experimenting with these knobs changes the balance between accuracy and smoothness and can be used to simulate different use-cases (e.g. faithful digitization vs. planning a safe, smooth robot trajectory).

---

## Possible Extensions

This repo is intentionally minimal, but the same pipeline could be extended to:

* 2D parametric splines for full stroke trajectories (instead of banded x-to-y curves).
* Stroke segmentation and ordering (to reconstruct the drawing sequence).
* Applying similar metrics to robot-arm trajectories or neural decoding outputs.
* Comparing other spline families (B-splines, Bézier curves) or non-spline models.

For now, the code serves as a small, self-contained example of how to:

1. Turn raster shapes into 1D centerlines.
2. Fit different spline models to them.
3. Quantitatively compare fidelity vs smoothness.
