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

# State Dynamics

# Snitch travels ellipse in XY plane centered at (9.144, 0, 15.24)
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

# Update position and velocity of the seeker based on previous state
def update_seeker_state(seeker_position, seeker_velocity, seeker_accel, dt):
    seeker_velocity += seeker_accel * dt
    seeker_position += seeker_velocity * dt

    seeker_state = np.concatenate((seeker_position, seeker_velocity), axis=0)

    return seeker_state

# PPN Guidance

# Determine desired seeker acceleration based on position and velocity of the snitch
# 3D PN Reference: https://en.wikipedia.org/wiki/Proportional_navigation#:~:text=Proportional%20navigation%20
def PN_guidance(snitch_state, seeker_state):
    N = 5  # Navigation Gain

    Rt = snitch_state[:3] # Snitch position
    Vt = snitch_state[3:] # Snitch velocity
    Rm = seeker_state[:3] # Seeker poisition
    Vm = seeker_state[3:] # Seeker velocity

    R_LOS = Rt - Rm # LOS
    Vr = Vt - Vm # Velocity of snitch relative to seeker

    R_dot_R = np.dot(R_LOS, R_LOS)

    if R_dot_R < 1e-6:
        R_dot_R = 1e-6  # Avoid divide-by-zero

    Omega = np.cross(R_LOS, Vr) / R_dot_R  # LOS rotation vector
    seeker_accel = N * np.cross(Vr, Omega) # PN acceleration

    # Clip to max acceleration
    seeker_accel_mag = np.linalg.norm(seeker_accel)
    if seeker_accel_mag > max_seeker_accel:
        seeker_accel = seeker_accel/seeker_accel_mag * max_seeker_accel

    return seeker_accel

# PID Controller

# Implement gimbal feedback control using seeker acceleration as an input
def PID_gimbal_control(snitch_state, seeker_state, max_gimbal, dt):
    integral_error = 0.0
    prev_error = 0.0

    optimal_PN_accel = PN_guidance(snitch_state, seeker_state)

    seeker_velocity = seeker_state[-3:] # Broom pointing axis
    
    if np.linalg.norm(seeker_velocity) < 1e-5:
        broom_axis = np.array([1.0, 0.0, 0.0]) # Default direction for small (likely initial) velocities

    else:
        broom_axis = seeker_velocity / np.linalg.norm(seeker_velocity)

    if np.linalg.norm(optimal_PN_accel) < 1e-5:
        desired_dir = broom_axis  # Do nothig if no need to accelerate

    else:
        desired_dir = optimal_PN_accel / np.linalg.norm(optimal_PN_accel)

    # Angular difference between current broom axis and desired direction
    angle_error = np.arccos(np.clip(np.dot(broom_axis, desired_dir), -1.0, 1.0))

    # Rotate about this axis to gimbal
    rotation_axis = np.cross(broom_axis, desired_dir)

    if np.linalg.norm(rotation_axis) < 1e-5:
        rotation_axis = np.array([0.0, 0.0, 1.0]) # Arbitrarily choose z-axis if the broom and desired axes are parallel or opposite

    else:
        rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)

    # PID Gains (Tunable)
    Kp = 3.0 # Proportional Gain (present error)
    Ki = 0.5 # Integral Gain (SS error)
    Kd = 0.1 # Derivative Gain (future error)
    
    integral_error += angle_error * dt # Accumulate error discretely via numerical integration
    derivative_error = (angle_error - prev_error) / dt # Discrete derivative error
    prev_error = angle_error # For the next timestep

    max_gimbal_rad = np.deg2rad(max_gimbal)

    gimbal_angle = Kp * angle_error + Ki * integral_error + Kd * derivative_error
    gimbal_angle = np.clip(gimbal_angle, -max_gimbal_rad, max_gimbal_rad) # Apply gimbal constraint

    # Rotate broom axis toward desired direction by gimbal steering angle
    rotation = R.from_rotvec(gimbal_angle * rotation_axis)
    gimbal_direction = rotation.apply(broom_axis)

    # Scale acceleration magnitude
    accel_magnitude = min(np.linalg.norm(optimal_PN_accel), max_seeker_accel)
    broom_accel = accel_magnitude * gimbal_direction

    gimbal_angle = np.rad2deg(gimbal_angle)

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
    if time >= 300 and radial_distance > 15.24:
        print(f'Seeker out of bounds at t = {time} s')
        break

    time += dt

capture_time = time
capture_position = seeker_position
capture_distance_to_snitch = distance

print(f'Distance to Capture: {capture_distance_to_snitch} m')
print(f'Seeker Position at Capture: {capture_position} m')

# Plots
snitch_list = np.array(snitch_position_list)
seeker_list = np.array(seeker_position_list)
distance_list = np.array(distance_list)
gimbal_angle_list = np.array(gimbal_angle_list)
time_list = np.array(time_list)

# 3D Trajectory
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(snitch_list[:, 0], snitch_list[:, 1], snitch_list[:, 2], label='Snitch', linewidth=1.5)
ax1.plot(seeker_list[:, 0], seeker_list[:, 1], seeker_list[:, 2], label='Seeker', linewidth=1.5)
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