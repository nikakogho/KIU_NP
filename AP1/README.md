## Overview

This project demonstrates two vector norms (ℓ1 and ℓ∞), their induced matrix norms, distances between random vectors and matrices, and 2-D visual “slices” (sections) of the corresponding unit balls.

## Norms Used

* Vector norms for x ∈ ℝ⁴
  • ‖x‖₁ = Σᵢ |xᵢ|
  • ‖x‖∞ = maxᵢ |xᵢ|
* Induced (subordinate) matrix norms for A ∈ ℝ²ˣ²
  • ‖A‖₁ = max over columns j of Σᵢ |aᵢⱼ|  (maximum column sum)
  • ‖A‖∞ = max over rows i of Σⱼ |aᵢⱼ|     (maximum row sum)

## What the Code Does

1. Implements reusable functions:
   * vector_norm for ℓ1 and ℓ∞
   * matrix_norm for the corresponding induced ‖·‖₁ and ‖·‖∞
   * vector_distance and matrix_distance as norms of differences
   * reshape_2x2 to map a length-4 vector to a 2×2 matrix (row-major)
2. Generates two random 4-vectors x and y, then reshapes them into 2×2 matrices A and B.
3. Computes distances:
   * Vector: d(x, y) = ‖x − y‖ with ℓ1 and ℓ∞
   * Matrix: D(A, B) = ‖A − B‖ with induced ‖·‖₁ and ‖·‖∞
4. Visualizes 2-D unit-ball sections:
   * Vector sections: fix x₃ and x₄ at a chosen center xᵣ, vary (x₁, x₂), and plot the region { x : ‖x − xᵣ‖ ≤ 1 } in the (x₁, x₂) plane
   * Matrix sections: fix the second row of a chosen center Aᵣ, vary (A₁₁, A₁₂), and plot the region { A : ‖A − Aᵣ‖ ≤ 1 } in the (A₁₁, A₁₂) plane
5. Saves four figures showing these sections.

## What the Plots Show

* Vector unit-ball sections (vary (x₁, x₂), fix x₃, x₄):
  • ℓ1 (vector_L1_section.png): a diamond centered at (xᵣ₁, xᵣ₂); boundary is |Δx₁| + |Δx₂| = 1
  • ℓ∞ (vector_Linf_section.png): an axis-aligned square centered at (xᵣ₁, xᵣ₂); boundary is max(|Δx₁|, |Δx₂|) = 1
  The shaded interior shows points with norm ≤ 1 relative to the center; a marker indicates the center itself.

* Matrix unit-ball sections (vary (A₁₁, A₁₂), fix second row of Aᵣ):
  • ‖·‖₁ (matrix_L1_section.png): an axis-aligned square; here D = A − Aᵣ has nonzero entries only in the first row, so column sums reduce to |d₁₁| and |d₁₂|, making ‖D‖₁ = max(|d₁₁|, |d₁₂|)
  • ‖·‖∞ (matrix_Linf_section.png): a diamond; with only the first row nonzero, the row sum is |d₁₁| + |d₁₂|, so ‖D‖∞ = |d₁₁| + |d₁₂|
  As in the vector case, the interior is the ≤ 1 region and the level-1 contour is drawn explicitly; the center (Aᵣ₁₁, Aᵣ₁₂) is marked.

## Outputs (at `generated_plots` folder)
* vector_L1_section.png
* vector_Linf_section.png
* matrix_L1_section.png
* matrix_Linf_section.png

## Key Takeaways

* In 2-D slices, ℓ1 produces diamonds and ℓ∞ produces squares for vectors.
* For induced matrix norms, when only the first row varies, ‖·‖₁ yields a square (max of column magnitudes) and ‖·‖∞ yields a diamond (sum across the active row).
* These shapes reflect how each norm aggregates deviations (sum vs maximum) and provide geometric intuition for stability and error analysis under different norms.
