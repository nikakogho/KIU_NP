import numpy as np

# Vector norms
def vector_norm(v: np.ndarray, ord: str = "1") -> float:
    """
    Compute vector l1 or linf norm.
    ord: "1" for l1, "inf" for linf
    """
    v = np.asarray(v).reshape(-1)
    if ord == "1":
        return float(np.sum(np.abs(v))) # sum of absolute values
    elif ord == "inf":
        return float(np.max(np.abs(v))) # maximum absolute value
    else:
        raise ValueError("Only '1' and 'inf' vector norms are supported in this demo.")

# Induced matrix norms
def matrix_norm(A: np.ndarray, ord: str = "1") -> float:
    """
    Compute induced matrix norm subordinate to l1 or linf:
      - ord=="1":   max column sum  (||A||_1)
      - ord=="inf": max row    sum  (||A||_inf)
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if ord == "1":
        return float(np.max(np.sum(np.abs(A), axis=0)))
    elif ord == "inf":
        return float(np.max(np.sum(np.abs(A), axis=1)))
    else:
        raise ValueError("Only '1' and 'inf' matrix norms are supported in this demo.")

# Distances
def vector_distance(x: np.ndarray, y: np.ndarray, ord: str = "1") -> float:
    """d(x,y) = ||x - y||_ord"""
    return vector_norm(np.asarray(x) - np.asarray(y), ord=ord)

def matrix_distance(A: np.ndarray, B: np.ndarray, ord: str = "1") -> float:
    """D(A,B) = ||A - B||_ord (induced norm matching ord)"""
    return matrix_norm(np.asarray(A) - np.asarray(B), ord=ord)

# Vector To Matrix
def reshape_2x2(v: np.ndarray) -> np.ndarray:
    """Reshape a length-4 vector into a 2x2 matrix (row-major)."""
    v = np.asarray(v).reshape(-1)
    if v.size != 4:
        raise ValueError("Expected a length-4 vector to reshape to 2x2.")
    return v.reshape(2, 2)

if __name__ == "__main__":
    v = np.array([1, -2, 3, -4])
    A = reshape_2x2(v)

    print("v =", v)
    print("A =\n", A)

    for ord in ("1", "inf"):
        print(f"\nVector norms (ord={ord}):")
        print("  ||v|| =", vector_norm(v, ord=ord))

        print(f"Matrix norms (ord={ord}):")
        print("  ||A|| =", matrix_norm(A, ord=ord))
