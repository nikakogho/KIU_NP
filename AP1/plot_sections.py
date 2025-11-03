# plot_sections.py
# Generates 4 2D unit-ball section plots.

import numpy as np
import matplotlib.pyplot as plt
from ap1_norm_distance import vector_norm, matrix_norm, reshape_2x2

from pathlib import Path
OUT_DIR = Path("generated_plots")
def savefig_safe(p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=150, bbox_inches="tight")

def grid(center_x, center_y, radius=1.2, n=400):
    xs = np.linspace(center_x - radius, center_x + radius, n)
    ys = np.linspace(center_y - radius, center_y + radius, n)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def vector_section_images(xr, ords=("1", "inf")):
    # Fix x3, x4 at xr[2], xr[3], vary (x1, x2)
    c1, c2 = xr[0], xr[1]
    X, Y = grid(c1, c2, radius=1.2, n=400)

    for ord in ords:
        # compute level = || [X-c1, Y-c2, 0, 0] ||_ord using vector_norm from norms_demo
        def level_val(x, y):
            v = np.array([x - c1, y - c2, 0.0, 0.0])
            return vector_norm(v, ord=ord)
        level = np.vectorize(level_val)(X, Y)
        mask = level <= 1.0

        plt.figure()
        plt.contourf(X, Y, mask.astype(float), levels=[0, 0.5, 1.0])
        plt.contour(X, Y, level, levels=[1.0])
        plt.gca().set_aspect("equal", "box")
        plt.scatter([c1], [c2])  # mark the center
        title = f"Vector section (x1,x2) with l{ord}"
        plt.title(title)
        plt.xlabel("x1"); plt.ylabel("x2")
        out = f"generated_plots/vector_L{ord}_section.png" if ord != "inf" else "generated_plots/vector_Linf_section.png"
        savefig_safe(out)
        plt.close()

def matrix_section_images(Ar, ords=("1", "inf")):
    # Fix second row at Ar[1,:], vary (a11, a12)
    a11c, a12c = Ar[0,0], Ar[0,1]
    X, Y = grid(a11c, a12c, radius=1.2, n=400)

    for ord in ords:
        # Build A per grid point with fixed second row, and compute ||A - Ar||_ord via matrix_norm
        def level_val(a11, a12):
            A = np.array([[a11, a12], [Ar[1,0], Ar[1,1]]])
            D = A - Ar
            return matrix_norm(D, ord=ord)
        level = np.vectorize(level_val)(X, Y)
        mask = level <= 1.0

        plt.figure()
        plt.contourf(X, Y, mask.astype(float), levels=[0, 0.5, 1.0])
        plt.contour(X, Y, level, levels=[1.0])
        plt.gca().set_aspect("equal", "box")
        plt.scatter([a11c], [a12c])  # mark the center
        title = f"Matrix section (a11,a12) with ||·||_{ord}"
        plt.title(title)
        plt.xlabel("a11"); plt.ylabel("a12")
        out = f"generated_plots/matrix_L{ord}_section.png" if ord != "inf" else "generated_plots/matrix_Linf_section.png"
        savefig_safe(out)
        plt.close()

def main():
    rng = np.random.default_rng(123)
    xr = rng.normal(size=4)     # center vector
    Ar = reshape_2x2(xr)        # center matrix (reshape same 4 numbers)

    vector_section_images(xr, ords=("1", "inf"))
    matrix_section_images(Ar, ords=("1", "inf"))
    print("Saved plots to generated_plots folder:")
    print(" - vector_L1_section.png")
    print(" - vector_Linf_section.png")
    print(" - matrix_L1_section.png")
    print(" - matrix_Linf_section.png")

if __name__ == "__main__":
    main()
