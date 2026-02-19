import numpy as np
import matplotlib.pyplot as plt
from navigation.dynamics import step_ivp_ext

def main():
    dt = 0.05
    steps = 600
    robot_r = 12.0
    ped_r = 14.0
    
    k_p = 16.0
    k_d = 8.0
    # Standard, simple radial force
    k_ped = 400.0  
    k_wall = 800.0
    v_max = 300.0
    R_safe = robot_r + ped_r + 40.0
    
    road_left = 20.0
    road_right = 180.0

    x = np.array([[100.0, 10.0]], dtype=np.float32)
    v = np.array([[0.0, 0.0]], dtype=np.float32)
    goal = np.array([100.0, 190.0], dtype=np.float32)
    
    # FIRST PRINCIPLES: REALISTIC TRAFFIC GAPS ---
    # Instead of a uniform wall, we group them into "platoons"
    ped_pos = np.array([
        [-20.0, 100.0],  # Platoon 1 (Left)
        [ 20.0, 100.0],
        # MASSIVE GAP HERE ---
        [150.0, 100.0],  # Platoon 2 (Right)
        [190.0, 100.0]
    ], dtype=np.float32)
    
    ped_vel = np.array([35.0, 0.0], dtype=np.float32) 
    K = len(ped_pos)
    
    traj_robot = []
    traj_peds = []
    collisions = 0
    
    print("Simulating cross-traffic with Realistic Gaps (Platoons)...")
    
    for t in range(steps):
        traj_robot.append(x[0].copy())
        traj_peds.append(ped_pos.copy())
        
        ped_pos += ped_vel * dt
        for k in range(K):
            if ped_pos[k, 0] > 220:
                ped_pos[k, 0] = -40
                
        dists = np.linalg.norm(ped_pos - x[0], axis=1)
        if np.any(dists < (robot_r + ped_r)):
            collisions += 1
            
        T_arr = np.array([goal], dtype=np.float32)
        
        # PURE, SIMPLE RADIAL REPULSION
        F_ped = np.zeros(2, dtype=np.float32)
        for p in ped_pos:
            diff = x[0] - p
            dist = float(np.linalg.norm(diff))
            if dist < R_safe and dist > 1e-3:
                n = diff / dist
                mag = k_ped * (R_safe - dist)
                F_ped += mag * n  # Pushes directly away, no fancy steering
                
        F_wall = np.zeros(2, dtype=np.float32)
        if x[0, 0] < road_left + robot_r:
            F_wall[0] += k_wall * ((road_left + robot_r) - x[0, 0])
        elif x[0, 0] > road_right - robot_r:
            F_wall[0] -= k_wall * (x[0, 0] - (road_right - robot_r))
        
        F_ext = np.array([F_ped + F_wall], dtype=np.float32)
        
        x, v = step_ivp_ext(
            x, v, T_arr, f_ext=F_ext,
            dt=dt, m=1.0, k_p=k_p, k_d=k_d, k_rep=0.0, R_safe=1.0, v_max=v_max
        )
        
        if np.linalg.norm(x[0] - goal) < 5.0:
            print(f"SUCCESS: Reached goal in {t} steps!")
            break

    traj_robot = np.array(traj_robot)
    traj_peds = np.array(traj_peds)
    
    plt.figure(figsize=(6, 6))
    plt.plot(traj_robot[:, 0], traj_robot[:, 1], 'g-', linewidth=2, label='Robot Trajectory')
    
    plt.axvline(road_left, color='black', linestyle='--', alpha=0.5, label='Road Edge')
    plt.axvline(road_right, color='black', linestyle='--', alpha=0.5)
    
    for k in range(K):
        plt.plot(traj_peds[:, k, 0], traj_peds[:, k, 1], 'r--', alpha=0.4)
        circ = plt.Circle(traj_peds[-1, k], ped_r, color='r', alpha=0.3)
        plt.gca().add_patch(circ)
        
    plt.scatter([100], [10], c='g', marker='o', s=100, label='Start')
    plt.scatter([100], [190], c='b', marker='X', s=100, label='Goal')
    
    plt.title(f"Realistic Gaps | Collisions: {collisions}")
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.gca().invert_yaxis() 
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_file = "runs/test_frogger_gaps.png"
    plt.savefig(out_file, dpi=120)
    print(f"Saved trajectory plot to {out_file}")

if __name__ == "__main__":
    main()