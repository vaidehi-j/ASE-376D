import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.transform import Rotation as R

class GolfBall:
    
    def __init__(self, initial_velocity, initial_velocity_tol, launch_angle, launch_angle_tol, aim_misalignment, aim_misalignment_tol, initial_spinrate, initial_spinrate_tol, spin_misalignment, spin_misalignment_tol):
        self.initial_velocity = stats.truncnorm.rvs(-3, 3, loc=initial_velocity, scale=initial_velocity_tol/3, size=1) 
        self.launch_angle = stats.truncnorm.rvs(-3, 3, loc=launch_angle, scale=launch_angle_tol/3, size=1)
        self.aim_misalignment = stats.truncnorm.rvs(-3, 3, loc=aim_misalignment, scale=aim_misalignment_tol/3, size=1)
        self.initial_spinrate = stats.truncnorm.rvs(-3, 3, loc=initial_spinrate, scale=initial_spinrate_tol/3, size=1)
        self.spin_misalignment = stats.truncnorm.rvs(-3, 3, loc=spin_misalignment, scale=spin_misalignment_tol/3, size=1)
    
    def get_trajectory(self, v0, aim_misalignment, launch_angle, dt):
        R = 1737.4 * 1000 # Lunar radius, [m]
        
        theta = np.deg2rad(aim_misalignment) # Azimuthal angle WRT +x-axis, derived from aiming misalignment [rad]
        phi = np.deg2rad(90-launch_angle) # Inclination angle WRT +z-axis, derived from launch angle [rad]

        # Velocity from spherical angles
        vx = float(v0 * np.sin(phi) * np.cos(theta))
        vy = float(v0 * np.sin(phi) * np.sin(theta))
        vz = float(v0 * np.cos(phi))

        # Position
        x = R
        y = 0
        z = 0
        r = np.array([x, y, z])
        
        g = 1.625 # Lunar gravitational acceleration [m/s^2]
        t = 0
        time = np.array([]) # Time vector [s] 
        
        position = np.empty((0, 3)) # Trajectory with x, y, z at each timestep [m]
        velocity = np.empty((0, 3)) # Velocity with x, y, z at each timestep [m/s]

        while np.linalg.norm(r) >= R: # Impact detection
            # Position and velocity of the previous timestep
            position = np.vstack((position, np.array([x, y, z]).reshape(1, 3)))
            velocity = np.vstack((velocity, np.array([vx, vy, vz]).reshape(1, 3)))

            dv = (-g * (r / np.linalg.norm(r))) * dt

            vx += dv[0]
            vy += dv[1]
            vz += dv[2]

            # Position at the current timestep
            x += vx * dt
            y += vy * dt
            z += vz * dt

            r = np.array([x, y, z])

            t += dt
            time = np.append(time, t)

        return position, velocity, time
    
    def get_orientation(self, spin_misalignment, initial_spinrate, dt, time):
        alpha = np.deg2rad(spin_misalignment) # Spin misalignment angle [rad]
        
        # Unit spin axis in body frame
        ux = float(np.sin(alpha))
        uy = 0.0
        uz = float(np.cos(alpha))

        omega = float(initial_spinrate) * 2*np.pi/60 # [rad/s]

        q = np.array([1.0, 0.0, 0.0, 0.0]) # Initial quaternion (no rotation)

        theta_spin_list = np.array([])
        quaternion = np.empty((0, 4)) # Quaternion with w, x, y, z components at each timestep 
        orientation = np.empty((0, 3)) # Orientation with x, y, z components at each timestep

        for i in range(len(time)):
            q0, q1, q2, q3 = q

            w1 = ux * omega
            w2 = uy * omega
            w3 = uz * omega

            q_matrix = np.array([[-q1, -q2, -q3],
                                 [q0, -q3, q2],
                                 [q3, q0, -q1],
                                 [-q2, q1, q0]
                                 ])
            
            w = np.transpose(np.array([w1, w2, w3]))

            q_dot = 0.5 * q_matrix @ w

            q = q + q_dot * dt
            q = q / np.linalg.norm(q)

            w, x, y, z = q
            
            quaternion = np.vstack((quaternion, np.array([w, x, y, z]).reshape(1, 4)))
            orientation = np.vstack((orientation, np.array([x, y, z]).reshape(1, 3)))

            theta_spin = np.arccos(q[0]) * 2
            theta_spin_list = np.append(theta_spin_list, theta_spin)

        return quaternion, orientation, theta_spin_list

def multiply_quaternions(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        q_new = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

        return q_new

def main():
    N = 500 # Number of test cases 
    num_bins = int(np.ceil(np.sqrt(N)))

    initial_velocity = 85 # [m/s]
    initial_velocity_tol = 5 # [m/s]

    launch_angle = 45 # [deg]
    launch_angle_tol = 2 # [deg]

    aim_misalignment = 0 # [deg]
    aim_misalignment_tol = 3 # [deg]

    initial_spinrate = 2000 # [rpm]
    initial_spinrate_tol = 500 # [rpm]

    spin_misalignment = 0 # [deg]
    spin_misalignment_tol = 3 # [deg]

    dt = 0.1 # [s]

    R_moon = 1737.4 * 1000 # [m]

    # Yarnball Plots
    fig1, ((ax1_1, ax1_2, ax1_3), (ax1_4, ax1_5, ax1_6), (ax1_7, ax1_8, ax1_9)) = plt.subplots(3, 3, figsize=(20, 20)) 
    ax1_1.grid(True)
    ax1_1.set_xlabel('Downrange Distance [m]')
    ax1_1.set_ylabel('Altitude [m]')
    ax1_2.grid(True)
    ax1_2.set_xlabel('Time [s]')
    ax1_2.set_ylabel('Quaternion $w$')
    ax1_3.grid(True)
    ax1_3.set_xlabel('Time [s]')
    ax1_3.set_ylabel('Quaternion $x$') 
    ax1_4.grid(True)
    ax1_4.set_xlabel('Time [s]')
    ax1_4.set_ylabel('Quaternion $y$') 
    ax1_5.grid(True)
    ax1_5.set_xlabel('Time [s]')
    ax1_5.set_ylabel('Quaternion $z$')
    ax1_6.grid(True)
    ax1_6.set_xlabel('Time [s]')
    ax1_6.set_ylabel('Roll [deg]')
    ax1_7.grid(True)
    ax1_7.set_xlabel('Time [s]')
    ax1_7.set_ylabel('Pitch [deg]')
    ax1_8.grid(True)
    ax1_8.set_xlabel('Time [s]')
    ax1_8.set_ylabel('Yaw [deg]')
    ax1_9.set_xlabel('Time [s]')
    ax1_9.set_ylabel('Total Spin Angle [deg]')
    ax1_9.grid(True)

    # Scatter Plots
    fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(3, 1, figsize=(7.5, 10)) 
    ax2_1.grid(True)
    ax2_1.set_xlabel('Longitude [deg]')
    ax2_1.set_ylabel('Latitude [deg]')
    ax2_2.grid(True)
    ax2_2.set_xlabel('Maximum Downrange Distance [m]')
    ax2_2.set_ylabel('Launch Angle [deg]')
    ax2_3.grid(True)
    ax2_3.set_xlabel('Maximum Downrange Distance [m]')
    ax2_3.set_ylabel('Aiming Misalignment [deg]')
    
    # 3D Trajectory Plot
    fig3 = plt.figure(figsize=(9, 6)) 
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_xlabel('$x$ [m]')
    ax3.set_ylabel('$y$ [m]')
    ax3.set_zlabel('$z$ [m]')

    fig4, ((ax4_1, ax4_2), (ax4_3, ax4_4), (ax4_5, ax4_6), (ax4_7, ax4_8)) = plt.subplots(4, 2, figsize=(20, 20)) # Histograms
    ax4_1.grid(True)
    ax4_1.set_xlabel('Initial Velocity [m/s]')
    ax4_1.set_ylabel('Count')
    ax4_2.grid(True)
    ax4_2.set_xlabel('Launch Angle [deg]')
    ax4_2.set_ylabel('Count')
    ax4_3.grid(True)
    ax4_3.set_xlabel('Aiming Misalignment [deg]')
    ax4_3.set_ylabel('Count')
    ax4_4.grid(True)
    ax4_4.set_xlabel('Spin Rate [rpm]')
    ax4_4.set_ylabel('Count')
    ax4_5.grid(True)
    ax4_5.set_xlabel('Spin Misalignment [deg]')
    ax4_5.set_ylabel('Count')
    ax4_6.grid(True)
    ax4_6.set_xlabel('Flight Duration [s]')
    ax4_6.set_ylabel('Count')
    ax4_7.grid(True)
    ax4_7.set_xlabel('Maximum Downrange Distance [m]')
    ax4_7.set_ylabel('Count')
    ax4_8.grid(True)
    ax4_8.set_xlabel('Maximum Alitude [m]')
    ax4_8.set_ylabel('Count')
    
    v0_list = phi_list = theta_list = spin_rate_list = alpha_list = duration_list = downrange_list = altitude_list = np.array([])
    
    for i in range(N):
        ball = GolfBall(initial_velocity=initial_velocity, 
                        initial_velocity_tol=initial_velocity_tol, 
                        launch_angle=launch_angle, 
                        launch_angle_tol=launch_angle_tol, 
                        aim_misalignment=aim_misalignment, 
                        aim_misalignment_tol=aim_misalignment_tol, 
                        initial_spinrate=initial_spinrate, 
                        initial_spinrate_tol=initial_spinrate_tol, 
                        spin_misalignment=spin_misalignment, 
                        spin_misalignment_tol=spin_misalignment_tol)
        
        position, velocity, time = ball.get_trajectory(v0=ball.initial_velocity, 
                                                  aim_misalignment=ball.aim_misalignment, 
                                                  launch_angle=ball.launch_angle, 
                                                  dt=dt)
    
        quaternion, orientation, theta_spin_list = ball.get_orientation(spin_misalignment=ball.spin_misalignment,
                                                                  initial_spinrate=ball.initial_spinrate,
                                                                  dt=dt,
                                                                  time=time)
        # Trajectory
        x_position = position[:, 0].reshape(len(time), 1)
        y_position = position[:, 1].reshape(len(time), 1)
        z_position = position[:, 2].reshape(len(time), 1) 
       
        downrange_distance = np.sqrt(y_position**2 + z_position**2)
        altitude = np.sqrt(x_position**2 + y_position**2 + z_position**2) - R_moon

        radial_distance = np.sqrt((x_position)**2 + (y_position)**2 + (z_position)**2)
        radial_distance = np.clip(radial_distance, 1e-10, np.inf)  # Avoid dividing by very small values
        latitude = np.rad2deg(np.arcsin((z_position)/radial_distance))
        longitude = np.rad2deg(np.arctan2(y_position, x_position))

        v0_list = np.append(v0_list, ball.initial_velocity)
        phi_list = np.append(phi_list, ball.launch_angle)
        theta_list = np.append(theta_list, ball.aim_misalignment)
        spin_rate_list = np.append(spin_rate_list, ball.initial_spinrate)
        alpha_list = np.append(alpha_list, ball.spin_misalignment)
        duration_list = np.append(duration_list, time[-1])
        downrange_list = np.append(downrange_list, np.max(downrange_distance))
        altitude_list = np.append(altitude_list, np.max(altitude))

        # Orientation
        w = quaternion[:, 0].reshape(len(time), 1)
        x = quaternion[:, 1].reshape(len(time), 1)
        y = quaternion[:, 2].reshape(len(time), 1)
        z = quaternion[:, 3].reshape(len(time), 1)  
        
        # Euler angles
        roll = np.atan2((2 * (w*x + y*z)), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(2 * (w*y - z*x))
        yaw = np.atan2(2 * (w*z + x*y), 1 - 2*(y**2 + z**2))

        # Plots
        # Yarnballs
        ax1_1.plot(downrange_distance, x_position - R_moon, alpha=0.5)
        ax1_2.plot(time, w, alpha=0.5)
        ax1_3.plot(time, x, alpha=0.5)
        ax1_4.plot(time, y, alpha=0.5)
        ax1_5.plot(time, z, alpha=0.5)
        ax1_6.plot(time, np.rad2deg(roll), alpha=0.5)
        ax1_7.plot(time, np.rad2deg(pitch), alpha=0.5)
        ax1_8.plot(time, np.rad2deg(yaw), alpha=0.5)
        ax1_9.plot(time, np.rad2deg(theta_spin_list), alpha=0.5)

        # Scatters
        ax2_1.scatter(longitude[-1], latitude[-1]) # landing latitude and longitude
        ax2_2.scatter(max(downrange_distance), ball.launch_angle)
        ax2_3.scatter(max(downrange_distance), ball.aim_misalignment)

        # 3D
        fig3.suptitle(f'3D Trajectory Visualization')
        ax3.plot(z_position, y_position, x_position - R_moon, alpha=0.5)
    
    # Histograms
    ax4_1.hist(v0_list, bins=num_bins)
    ax4_2.hist(phi_list, bins=num_bins)
    ax4_3.hist(theta_list, bins=num_bins)
    ax4_4.hist(spin_rate_list, bins=num_bins)
    ax4_5.hist(alpha_list, bins=num_bins)
    ax4_6.hist(duration_list, bins=num_bins)
    ax4_7.hist(downrange_list, bins=num_bins)
    ax4_8.hist(altitude_list, bins=num_bins)


    plt.show()

if __name__ == '__main__':
    main()