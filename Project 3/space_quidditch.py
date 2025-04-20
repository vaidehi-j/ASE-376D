import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.spatial.transform import Rotation as R

# Seeker parameters
seeker_mass = 100 # [kg]
max_seeker_accel = 1.5 * 9.81 # [m/s^2]
max_gimbal = 50 # [deg]
seeker_position = np.array([0.0, 0.0, 0.0]) # initialial position [m]
seeker_velocity = np.array([0.0, 0.0, 0.0]) # Initial velocity [m/s]

seeker_position = seeker_position.astype(float)
seeker_velocity = seeker_velocity.astype(float)

# Snitch Parameters
a = 9.144 # Semimajor axis [m]
b = 1.524 # Semiminor axis [m]
ellipse_center = np.array([9.144, 0, 15.24]) # [m]
T_snitch = 5 * 60 # Period [s]
snitch_mass = 0.2 # [kg]

max_sim_time = 300 # [s]
capture_distance = 0.3 # [m] 
dt = 0.1 # [s]

def get_snitch_state(t):
    theta = 2*np.pi*t / T_snitch # Angle swept by snitch [rad]
    
    x = ellipse_center[0] + a*np.cos(theta)
    y = ellipse_center[1] + b*np.sin(theta)
    z = ellipse_center[2] 

    # Derivatives of x, y, z position
    vx = -a * np.sin(theta) * (2*np.pi / T_snitch)
    vy = b * np.cos(theta) * (2*np.pi / T_snitch)
    vz = 0

    snitch_position = np.array([x, y, z])
    snitch_velocity = np.array([vx, vy, vz])

    snitch_state = np.concatenate((snitch_position, snitch_velocity), axis=0)

    return snitch_state

def update_seeker_state(seeker_position, seeker_velocity, seeker_accel, dt):
    seeker_velocity += seeker_accel * dt
    seeker_position += seeker_velocity * dt

    seeker_state = np.concatenate((seeker_position, seeker_velocity), axis=0)

    return seeker_state

def PN_guidance(snitch_state, seeker_state):
    N = 5 # Tunable navigation gain
    
    xt, yt, zt = snitch_state[:3]
    vtx, vty, vtz = snitch_state[-3:]

    xm, ym, zm = seeker_state[:3]
    vmx, vmy, vmz = seeker_state[-3:]
    
    vm = np.sqrt(vmx**2 + vmy**2 + vmz**2) 

    lambda_angle = np.atan2((yt-ym), (xt-xm))
    lambda_dot = ((xt-xm)*(vty-vmy) - (yt-ym)*(vtx-vmx)) / ((xt-xm)**2 + (yt-ym)**2) 

    lateral_accel = N * vm * lambda_dot # Acceleration perpendicular to snitch velocity [m/s^2]
    
    ax = -lateral_accel * np.cos(lambda_angle)
    ay = lateral_accel * np.sin(lambda_angle)

    z_error = zt - zm
    z_dot_error = vtz - vmz
    az = 1.5 * z_error + 0.5 *z_dot_error # PD for z-accel in particular

    seeker_accel = np.array([ax, ay, az])
 
    if np.linalg.norm(seeker_accel) > max_seeker_accel:
        seeker_accel = seeker_accel / np.linalg.norm(seeker_accel) * max_seeker_accel * np.sign(lateral_accel)

    return seeker_accel

def LQR_gimbal_control(snitch_state, seeker_state, max_gimbal, dt):
    PN_accel = PN_guidance(snitch_state, seeker_state)

    # A, B matrices derive from kinematics (system definition)
    A_sys = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]
                ])
    
    B_sys = np.array([[0.5*dt**2, 0, 0],
                  [0, 0.5*dt**2, 0],
                  [0, 0, 0.5*dt**2],
                  [dt, 0, 0],
                  [0, dt, 0],
                  [0, 0, dt]
                ])
    
    # Q, R are cost matrices (state/control penalties)
    Q_cost = np.diag([200, 200, 200, 20, 20, 20])
    R_cost = np.diag([5, 5, 5])
    # Q_cost = np.diag([100, 100, 100, 10, 10, 10]) # Tunable parameter
    # R_cost = np.eye(3)

    # Gain Matrix K to minimize cost function
    K, __, __ = ct.lqr(A_sys, B_sys, Q_cost, R_cost)

    e = seeker_state - snitch_state # Control input depends on error between full state dynamics
    LQR_accel = -K @ e # Optimal acceleration as dictated by linear feedback law

    broom_accel = PN_accel + LQR_accel 
    
    # At this stage, may violate gimbal constraint

    seeker_velocity = seeker_state[-3:]
    if np.linalg.norm(seeker_velocity) < 1e-5:
        broom_axis = np.array([1, 0, 0]) # Default

    else:
        broom_axis = seeker_velocity / np.linalg.norm(seeker_velocity) # Broom axis is the direction of seeker velocity


    # Apply gimbal constraint
    broom_accel_norm = np.linalg.norm(broom_accel)

    if broom_accel_norm > 1e-5: # check for zero acceleration
        
        broom_accel_dir = broom_accel / broom_accel_norm
        gimbal_steer_angle = np.arccos(np.clip(np.dot(broom_accel_dir, broom_axis), -1.0, 1.0)) # Angle between 

        max_angle_rad = np.deg2rad(max_gimbal)

        if gimbal_steer_angle > max_angle_rad:
            
            rotation_axis = np.cross(broom_axis, broom_accel_dir) # Rotate broom axis toward commanded acceleration by max gimbal angle


            if np.linalg.norm(rotation_axis) < 1e-5:
                rotation_axis = np.array([0, 0, 1]) # Arbitrary axis if broom and acceleration axes are approx parallel

            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            rotation = R.from_rotvec(max_angle_rad * rotation_axis) 
            allowed_accel_dir = rotation.apply(broom_axis)

            broom_accel = allowed_accel_dir * min(broom_accel_norm, max_seeker_accel)

        else:
            broom_accel = broom_accel * min(1.0, max_seeker_accel / broom_accel_norm)

    else:
        broom_accel = np.zeros(3)

    gimbal_angle = np.rad2deg(gimbal_steer_angle)

    return broom_accel, gimbal_angle

# Main Simulation Framework

time = 0.0
snitch_position_list = []
seeker_position_list = []
distance_list = []
gimbal_angle_list = []
time_list = []

seeker_initial_position = np.array([0.0, 0.0, 0.0])
seeker_initial_velocity = np.array([0.0, 0.0, 0.0])

# Initial state
seeker_position = seeker_initial_position.copy()
seeker_velocity = seeker_initial_velocity.copy()

while time < max_sim_time:
    snitch_state = get_snitch_state(time)
    
    seeker_state = np.concatenate((seeker_position, seeker_velocity), axis=0)

    seeker_accel, gimbal_angle = PID_gimbal_control(snitch_state, seeker_state, max_gimbal, dt)
    
    seeker_state = update_seeker_state(seeker_position, seeker_velocity, seeker_accel, dt)
    seeker_position = seeker_state[:3]
    seeker_velocity = seeker_state[3:]
    
    time_list.append(time)
    snitch_position_list.append(snitch_state[:3])
    seeker_position_list.append(seeker_position)
    distance = np.linalg.norm(snitch_state[:3] - seeker_position)
    distance_list.append(distance)
    gimbal_angle_list.append(gimbal_angle)
    
    # Capture Condition
    if distance <= capture_distance:
        print(f'Snitch captured at t = {time} s')
        break

    # Out of Boudns Condition
    radial_distance = np.linalg.norm(seeker_position - ellipse_center)

    if time > 250 and radial_distance > 15.24:
        print(f'Seeker out of bounds at t = {time} s')
        break

    time += dt

capture_time = time
capture_position = seeker_position
capture_distance_to_snitch = distance

# Plots

snitch_list = np.array(snitch_position_list)
seeker_list = np.array(seeker_position_list)
distance_list = np.array(distance_list)
gimbal_angle_list = np.array(gimbal_angle_list)
time_list = np.array(time_list)

print("Snitch shape:", snitch_list.shape)
print("Seeker shape:", seeker_list.shape)

# 3D Trajectory
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(snitch_list[:, 0], snitch_list[:, 1], snitch_list[:, 2], 'o-', label='Snitch', linewidth=2)
ax1.plot(seeker_list[:, 0], seeker_list[:, 1], seeker_list[:, 2], '-', label='Seeker', linewidth=2)
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]')
ax1.set_zlabel('$z$ [m]')
ax1.set_title('3D Trajectory of Snitch and Seeker')
ax1.legend()
ax1.grid()

# Distance to Snitch vs Time
fig2 = plt.figure(figsize=(8, 6))
plt.plot(time_list, distance_list, label='Distance to Snitch')
plt.axhline(y=capture_distance, color='red', linestyle='--', label='Capture Distance')
plt.xlabel('Time [s]')
plt.ylabel('Distance [m]')
plt.title('Distance to Snitch vs. Time')
plt.legend()
plt.grid()

# Gimbal Angle vs Time
fig3 = plt.figure(figsize=(8, 6))
plt.plot(time_list, gimbal_angle_list, label='Gimbal Angle')
plt.axhline(y=max_gimbal, color='red', linestyle='--', label='Max Gimbal')
plt.axhline(y=-max_gimbal, color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Gimbal Angle [deg]')
plt.title('Gimbal Angle vs. Time')
plt.legend()
plt.grid()

plt.show()