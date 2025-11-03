import numpy as np
from ap1_norm_distance import reshape_2x2, vector_distance, matrix_distance

def main(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=4)
    y = rng.normal(size=4)
    A = reshape_2x2(x)
    B = reshape_2x2(y)

    print("x =", x)
    print("y =", y)
    print("A =\n", A)
    print("B =\n", B)

    for ord in ("1", "inf"):
        dv = vector_distance(x, y, ord=ord)
        dm = matrix_distance(A, B, ord=ord)
        print(f"\nord={ord}:")
        print("  ||x - y|| =", dv)
        print("  ||A - B|| =", dm)

if __name__ == "__main__":
    main()
