import numpy as np
from navigation.dynamics import sat_velocity, repulsion_forces, step_ivp


def test_sat_velocity_never_exceeds_vmax():
    v = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = sat_velocity(v, v_max=2.0)
    norms = np.linalg.norm(out, axis=1)
    assert norms[0] <= 2.0 + 1e-9
    assert norms[1] == 0.0


def test_repulsion_is_equal_and_opposite_for_two_robots():
    x = np.array([[0.0, 0.0], [0.2, 0.0]])
    f = repulsion_forces(x, k_rep=1.0, R_safe=1.0)
    assert np.linalg.norm(f[0] + f[1]) < 1e-9
    assert f[0, 0] < 0
    assert f[1, 0] > 0


def test_step_ivp_moves_toward_target_when_no_repulsion():
    x = np.array([[0.0, 0.0]])
    v = np.array([[0.0, 0.0]])
    T = np.array([[10.0, 0.0]])

    x2, v2 = step_ivp(
        x, v, T,
        dt=0.1,
        m=1.0,
        k_p=1.0,
        k_d=0.0,
        k_rep=0.0,
        R_safe=1.0,
        v_max=100.0,
    )
    assert x2[0, 0] > x[0, 0]
