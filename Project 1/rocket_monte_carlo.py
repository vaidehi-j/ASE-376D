import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class Rocket:

    # Create a rocket object with properties corresponding to the follwing input dispersions
    def __init__(self, thrust, thrust_tolerance, dry_mass, dry_mass_tolerance, wet_mass, wet_mass_tolerance, mixture_ratio, mixture_ratio_tolerance, burn_duration, burn_duration_tolerance, dt):
        # Normal distributions are within three STD DEV of the mean
        self.thrust = (stats.truncnorm.rvs(-3, 3, loc=thrust, scale=thrust_tolerance, size=1)) * 1000 # Normal, scaled from [kN] to [N]
        self.dry_mass = dry_mass + np.random.uniform(low=-dry_mass_tolerance, high=dry_mass_tolerance, size=1) # Uniform
        self.wet_mass = stats.truncnorm.rvs(-3, 3, loc=wet_mass, scale = wet_mass_tolerance, size=1) # Normal
        self.mixture_ratio = mixture_ratio + np.random.uniform(low=-mixture_ratio_tolerance, high=mixture_ratio_tolerance, size=1) # Uniform
        self.burn_duration = stats.truncnorm.rvs(-3, 3, loc=burn_duration, scale = burn_duration_tolerance, size=1) # Nomral
        self.initial_accel = 9.81 * self.mixture_ratio ** 0.4
        self.time = np.array(np.arange(0, self.burn_duration+dt, dt)) # Time vector to track ox and fuel depletion

    # Determine state of the rocket at liftoff
    def get_initial_state(self, thrust, wet_mass, mixture_ratio, initial_accel):
        initial_fuel = wet_mass / (mixture_ratio+1)
        initial_ox = initial_fuel * mixture_ratio
        initial_mass = thrust / initial_accel

        return np.array([initial_ox, initial_fuel, initial_mass])
    
    # Determine state of the rocket at end of burn
    def get_final_state(self, thrust, dry_mass, payload_mass):
        final_wet_mass = 0
        final_mass = final_wet_mass + payload_mass + dry_mass

        return final_mass
    
    # Calculate ox and fuel flow rates 
    def get_flow_rates(self, initial_ox, final_ox, initial_fuel, final_fuel, burn_duration):
        flowrate_ox = (initial_ox-final_ox) / burn_duration
        flowrate_fuel = (initial_fuel-final_fuel) / burn_duration

        return np.array([flowrate_ox, flowrate_fuel])
    
    # Calculate ox and fuel depletion WRT time
    def get_ox_mass(self, initial_ox, ox_flowrate, time):
        ox_mass = initial_ox - ox_flowrate * time

        return ox_mass
    
    def get_fuel_mass(self, initial_fuel, fuel_flowrate, time):
        fuel_mass = initial_fuel - fuel_flowrate * time

        return fuel_mass
    
    # Determine mass-to-orbit achievable by the rocket
    def get_payload_mass(self, initial_mass, dry_mass, wet_mass): 
        payload_mass = initial_mass - dry_mass - wet_mass

        return payload_mass

def main():
    N = 500 # Number of test cases
    num_bins = int(np.ceil(np.sqrt(N)))

    thrust = 10000 # [kN]
    thrust_tolerance = 200 # [kN]

    dry_mass = 150000 # [kg]
    dry_mass_tolerance = 200 # [kg]

    wet_mass = 500000 # [kg]
    wet_mass_tolerance = 500 # [kg]

    mixture_ratio = 2.5 # dimensionless
    mixture_ratio_tolerance = 0.1 # dimensionless

    burn_duration = 180 # [s]
    burn_duration_tolerance = 5 # [s]
    
    dt = 0.1 # [s]

    thrust_list = dry_mass_list = wet_mass_list = mixture_ratio_list = burn_duration_list = liftoff_ox_list = liftoff_fuel_list = final_ox_list = final_fuel_list = flowrate_ox_list = flowrate_fuel_list = payload_mass_list = np.array([])

    fig1, (ax1, ax2) = plt.subplots(2, 1) # Yarnball Plots
    ax1.set_xlabel('Burn Duration [s]')
    ax1.set_ylabel('Ox Mass [kg]')
    ax2.set_xlabel('Burn Duration [s]')
    ax2.set_ylabel('Fuel Mass [kg]')

    fig1.suptitle(f'Ox and Fuel Mass vs. Burn Duration for {N} Rockets')
   
    # Conduct 'N' Monte Carlo simulations
    for i in range(N):
        vehicle = Rocket(thrust=thrust, thrust_tolerance=thrust_tolerance, dry_mass=dry_mass, dry_mass_tolerance=dry_mass_tolerance, wet_mass=wet_mass, wet_mass_tolerance=wet_mass_tolerance, mixture_ratio=mixture_ratio, mixture_ratio_tolerance=mixture_ratio_tolerance, burn_duration=burn_duration, burn_duration_tolerance=burn_duration_tolerance, dt=dt)

        initial_state = vehicle.get_initial_state(thrust=vehicle.thrust, wet_mass=vehicle.wet_mass, mixture_ratio=vehicle.mixture_ratio, initial_accel=vehicle.initial_accel)
        initial_ox = initial_state[0]
        initial_fuel = initial_state[1]
        initial_mass = initial_state[-1]
        
        payload_mass = vehicle.get_payload_mass(initial_mass=initial_mass, dry_mass=vehicle.dry_mass, wet_mass=vehicle.wet_mass)

        final_state = vehicle.get_final_state(thrust=vehicle.thrust, dry_mass=vehicle.dry_mass, payload_mass=payload_mass)
        final_ox = 0
        final_fuel = 0
        final_mass = final_state[0]

        flow_rates = vehicle.get_flow_rates(initial_ox=initial_ox, final_ox=final_ox, initial_fuel=initial_fuel, final_fuel=final_fuel, burn_duration=vehicle.burn_duration)
        ox_flowrate = flow_rates[0]
        fuel_flowrate = flow_rates[-1]


        thrust_list = np.append(thrust_list, vehicle.thrust)
        dry_mass_list = np.append(dry_mass_list, vehicle.dry_mass)
        wet_mass_list = np.append(wet_mass_list, vehicle.wet_mass)
        mixture_ratio_list = np.append(mixture_ratio_list, vehicle.mixture_ratio)
        burn_duration_list = np.append(burn_duration_list, vehicle.burn_duration)

        liftoff_ox_list = np.append(liftoff_ox_list, initial_ox)
        liftoff_fuel_list = np.append(liftoff_fuel_list, initial_fuel)
        final_ox_list = np.append(final_ox_list, final_ox)
 
        flowrate_ox_list = np.append(flowrate_ox_list, ox_flowrate)
        flowrate_fuel_list = np.append(flowrate_fuel_list, fuel_flowrate)

        ox_mass = vehicle.get_ox_mass(initial_ox=initial_ox, ox_flowrate=ox_flowrate, time=vehicle.time)
        fuel_mass = vehicle.get_fuel_mass(initial_fuel=initial_fuel, fuel_flowrate=fuel_flowrate, time=vehicle.time)

        ax1.plot(vehicle.time, ox_mass, alpha=0.5)
        ax2.plot(vehicle.time, fuel_mass, alpha=0.5)

        payload_mass_list = np.append(payload_mass_list, payload_mass)
    
    # Data Visualization 
    fig2, ((ax11, ax22), (ax33, ax44)) = plt.subplots(2,2) # Input Parameter Histograms
    ax11.hist(thrust_list, bins=num_bins)
    ax11.set_xlabel('Thrust [N]')
    ax11.set_ylabel('Count')
    mu_thrust = np.mean(thrust_list)
    sigma_thrust = np.std(thrust_list, axis=0)
    ax11.axvline(mu_thrust, color='k', linestyle='dotted', linewidth=1) # Mean
    for i in range(3): # Three STD DEV on either side of the mean
        ax11.axvline(mu_thrust + (i+1)*sigma_thrust, color='r', linestyle='dotted', linewidth=1)
        ax11.axvline(mu_thrust - (i+1)*sigma_thrust, color='r', linestyle='dotted', linewidth=1)


    ax22.hist(dry_mass_list, bins=num_bins)
    ax22.set_xlabel('Dry Mass [kg]')
    ax22.set_ylabel('Count')
    mu_dry_mass = np.mean(dry_mass_list)
    sigma_dry_mass = np.std(dry_mass_list)
    ax22.axvline(mu_dry_mass, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax22.axvline(mu_dry_mass + (i+1)*sigma_dry_mass, color='r', linestyle='dotted', linewidth=1)
        ax22.axvline(mu_dry_mass - (i+1)*sigma_dry_mass, color='r', linestyle='dotted', linewidth=1)

    ax33.hist(mixture_ratio_list, bins=num_bins)
    ax33.set_xlabel('Mixture Ratio')
    ax33.set_ylabel('Count')
    mu_mixture_ratio = np.mean(mixture_ratio_list)
    sigma_mixture_ratio = np.std(mixture_ratio_list)
    ax33.axvline(mu_mixture_ratio, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax33.axvline(mu_mixture_ratio + (i+1)*sigma_mixture_ratio, color='r', linestyle='dotted', linewidth=1)
        ax33.axvline(mu_mixture_ratio - (i+1)*sigma_mixture_ratio, color='r', linestyle='dotted', linewidth=1)

    ax44.hist(burn_duration_list, bins=num_bins)
    ax44.set_xlabel('Burn Duration [s]')
    ax44.set_ylabel('Count')
    mu_burn_duration = np.mean(burn_duration_list)
    sigma_burn_duration = np.std(burn_duration_list)
    ax44.axvline(mu_burn_duration, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax44.axvline(mu_burn_duration + (i+1)*sigma_burn_duration, color='r', linestyle='dotted', linewidth=1)
        ax44.axvline(mu_burn_duration - (i+1)*sigma_burn_duration, color='r', linestyle='dotted', linewidth=1)

    fig2.suptitle(f'Input Dispersions for {N} Rockets')

    fig3, (ax111, ax222) = plt.subplots(2, 1) # Initial and Final Ox/Fuel Mass Histograms
    ax111.hist(liftoff_ox_list, bins=num_bins)
    ax111.set_xlabel('Liftoff Ox Mass [kg]')
    ax111.set_ylabel('Count')
    mu_liftoff_ox = np.mean(liftoff_ox_list)
    sigma_liftoff_ox = np.std(liftoff_ox_list, axis=0)
    ax111.axvline(mu_liftoff_ox, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax111.axvline(mu_liftoff_ox + (i+1)*sigma_liftoff_ox, color='r', linestyle='dotted', linewidth=1)
        ax111.axvline(mu_liftoff_ox - (i+1)*sigma_liftoff_ox, color='r', linestyle='dotted', linewidth=1)
 

    ax222.hist(liftoff_fuel_list, bins=num_bins)
    ax222.set_xlabel('Liftoff Fuel Mass [kg]')
    ax222.set_ylabel('Count')
    mu_liftoff_fuel = np.mean(liftoff_fuel_list)
    sigma_liftoff_fuel = np.std(liftoff_fuel_list, axis=0)
    ax222.axvline(mu_liftoff_fuel, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax222.axvline(mu_liftoff_fuel + (i+1)*sigma_liftoff_fuel, color='r', linestyle='dotted', linewidth=1)
        ax222.axvline(mu_liftoff_fuel - (i+1)*sigma_liftoff_fuel, color='r', linestyle='dotted', linewidth=1)


    fig3.suptitle(f'Liftoff Ox/Fuel Masses for {N} Rockets')

    fig4, (ax1111, ax2222) = plt.subplots(2, 1)
    ax1111.hist(flowrate_ox_list, bins=num_bins)
    ax1111.set_xlabel('Ox Mass Flowrate [kg/s]')
    ax1111.set_ylabel('Count')
    mu_ox_flowrate = np.mean(flowrate_ox_list)
    sigma_ox_flowrate = np.std(flowrate_ox_list)
    ax1111.axvline(mu_ox_flowrate, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax1111.axvline(mu_ox_flowrate + (i+1)*sigma_ox_flowrate, color='r', linestyle='dotted', linewidth=1)
        ax1111.axvline(mu_ox_flowrate - (i+1)*sigma_ox_flowrate, color='r', linestyle='dotted', linewidth=1)

    ax2222.hist(flowrate_fuel_list, bins=num_bins)
    ax2222.set_xlabel('Fuel Mass Flowrate [kg/s]')
    ax2222.set_ylabel('Count')
    mu_fuel_flowrate = np.mean(flowrate_fuel_list)
    sigma_fuel_flowrate = np.std(flowrate_fuel_list)
    ax2222.axvline(mu_fuel_flowrate, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax2222.axvline(mu_fuel_flowrate + (i+1)*sigma_fuel_flowrate, color='r', linestyle='dotted', linewidth=1)
        ax2222.axvline(mu_fuel_flowrate - (i+1)*sigma_fuel_flowrate, color='r', linestyle='dotted', linewidth=1)
 

    fig4.suptitle(f'Ox and Fuel Flowrates for {N} Rockets')

    fig5, ax5 = plt.subplots(1) # Payload Mass Histogram
    ax5.hist(payload_mass_list, bins=num_bins)
    ax5.set_xlabel('Payload Mass [kg]')
    ax5.set_ylabel('Count')
    mu_payload_mass = np.mean(payload_mass_list)
    sigma_payload_mass = np.std(payload_mass_list)
    ax5.axvline(mu_payload_mass, color='k', linestyle='dotted', linewidth=1)
    for i in range(3):
        ax5.axvline(mu_payload_mass + (i+1)*sigma_payload_mass, color='r', linestyle='dotted', linewidth=1)
        ax5.axvline(mu_payload_mass - (i+1)*sigma_payload_mass, color='r', linestyle='dotted', linewidth=1)
    print(mu_payload_mass)
    print(sigma_payload_mass)

    fig5.suptitle(f'Mass to Orbit for {N} Rockets')
    plt.show()

    # # Verify that all mass-to-orbit values lie within 3 STD DEV of the mean
    # Z_score_list = (payload_mass_list - mu_payload_mass) / sigma_payload_mass
    # print(f'Max STD DEV: {Z_score_list.max()} \n Min STD DEV: {Z_score_list.min()}')

if __name__ == "__main__":
    main()