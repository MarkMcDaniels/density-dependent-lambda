import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants (in SI units)
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
H0 = 70.0 * 1000 / 3.086e22  # Hubble constant in s^-1
Lambda0 = 3 * H0**2  # Background cosmological constant
Mpc_to_m = 3.086e22  # Megaparsec to meters
solar_mass = 1.989e30  # Solar mass in kg
kpc_to_m = 3.086e19  # Kiloparsec to meters

# Pinning model parameter (Convert from M_sun^-1 kpc^3 to SI units)
alpha = 0.05 / (solar_mass * kpc_to_m**-3)

print(f"Constants:")
print(f"G = {G:.4e} m^3 kg^-1 s^-2")
print(f"H0 = {H0:.4e} s^-1")
print(f"Lambda0 = {Lambda0:.4e} s^-2")
print(f"alpha = {alpha:.4e} m^3 kg^-1")

# Simple density-dependent lambda function
def lambda_pinning(density):
    """Calculate density-dependent cosmological 'constant'"""
    return Lambda0 * np.exp(-alpha * density)

# Get effective lambda including correction term
def lambda_effective(density):
    """Calculate effective cosmological constant with correction"""
    lambda_rho = lambda_pinning(density)
    return lambda_rho * (1 - alpha * density / 2)

# Test the lambda functions with sample densities
# Critical density of the universe
rho_crit = 3 * H0**2 / (8 * np.pi * G)

print("\nLambda values at different densities:")
densities = [0.1 * rho_crit, rho_crit, 10 * rho_crit, 100 * rho_crit]
for rho in densities:
    lambda_val = lambda_pinning(rho)
    lambda_eff = lambda_effective(rho)
    print(f"Density: {rho/rho_crit:.1f}×ρ_crit → λ(ρ): {lambda_val/Lambda0:.6f}×λ₀, λ_eff: {lambda_eff/Lambda0:.6f}×λ₀")

# Define the 1D collapse ODE system
def spherical_collapse(y, t, model):
    """ODE system for spherical collapse under different models"""
    r, v = y  # Radius and velocity
    
    # Initial conditions (fixed for all models)
    initial_radius = 10.0 * Mpc_to_m  # 10 Mpc
    overdensity = 1.1  # 10% overdensity
    initial_density = rho_crit * overdensity
    
    # Calculate enclosed mass
    mass = (4/3) * np.pi * initial_density * initial_radius**3
    
    # Current density (assuming mass conservation)
    if r > 0:
        density = mass / ((4/3) * np.pi * r**3)
    else:
        density = initial_density  # Avoid division by zero
    
    # Calculate acceleration based on model
    if model == "standard":
        # Standard Newtonian gravity only
        a = -G * mass / (r**2) if r > 0 else 0
        
    elif model == "lambda":
        # ΛCDM: Newtonian + constant λ
        a_newton = -G * mass / (r**2) if r > 0 else 0
        a_lambda = Lambda0 * r / 3  # Repulsive cosmological constant
        a = a_newton + a_lambda
        
    elif model == "pinning":
        # Pinning model: Newtonian + density-dependent λ
        a_newton = -G * mass / (r**2) if r > 0 else 0
        
        lambda_eff = lambda_effective(density)
        a_lambda = lambda_eff * r / 3
        
        # Additional pinning effect (correction from density dependence)
        # This term is unique to the pinning model
        pinning_correction = alpha * lambda_pinning(density) * density * r / 6
        
        a = a_newton + a_lambda + pinning_correction
    
    return [v, a]

# Run the simulations
def run_simulations():
    # Initial conditions
    initial_radius = 10.0 * Mpc_to_m  # 10 Mpc
    initial_velocity = H0 * initial_radius  # Hubble flow velocity
    y0 = [initial_radius, initial_velocity]
    
    # Time span (15 billion years in seconds)
    t_max = 15e9 * 365.25 * 24 * 3600
    t = np.linspace(0, t_max, 1000)
    
    results = {}
    
    # Run each model
    for model in ["standard", "lambda", "pinning"]:
        print(f"\nRunning {model} model...")
        
        # Integrate the ODEs
        solution = odeint(spherical_collapse, y0, t, args=(model,))
        
        # Extract radius and velocity
        radius = solution[:, 0]
        velocity = solution[:, 1]
        
        # Store results
        results[model] = {
            "time_Gyr": t / (1e9 * 365.25 * 24 * 3600),
            "radius_Mpc": radius / Mpc_to_m,
            "velocity_kms": velocity / 1000
        }
        
        # Print key statistics
        final_radius = radius[-1] / Mpc_to_m
        final_velocity = velocity[-1] / 1000
        print(f"  Final radius: {final_radius:.4f} Mpc")
        print(f"  Final velocity: {final_velocity:.4f} km/s")
        
        # Check for turnaround (when expansion stops)
        velocity_changes = np.diff(np.signbit(velocity))
        turnaround_indices = np.where(velocity_changes)[0]
        
        if len(turnaround_indices) > 0:
            turnaround_time = t[turnaround_indices[0]] / (1e9 * 365.25 * 24 * 3600)
            turnaround_radius = radius[turnaround_indices[0]] / Mpc_to_m
            print(f"  Turnaround at t = {turnaround_time:.2f} Gyr, r = {turnaround_radius:.2f} Mpc")
        else:
            print("  No turnaround detected")
    
    return results

# Visualize the results
def visualize_results(results):
    plt.figure(figsize=(15, 10))
    
    # Colors and styles
    styles = {
        "standard": {"color": "blue", "linestyle": "-", "label": "Standard Newtonian"},
        "lambda": {"color": "green", "linestyle": "--", "label": "ΛCDM Model"},
        "pinning": {"color": "red", "linestyle": "-", "label": "Pinning Model"}
    }
    
    # Plot radius evolution
    plt.subplot(2, 1, 1)
    for model, data in results.items():
        plt.plot(
            data["time_Gyr"],
            data["radius_Mpc"],
            color=styles[model]["color"],
            linestyle=styles[model]["linestyle"],
            linewidth=2.5,
            label=styles[model]["label"]
        )
    
    plt.title("Spherical Overdensity Evolution", fontsize=16)
    plt.xlabel("Time (Gyr)", fontsize=14)
    plt.ylabel("Radius (Mpc)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Plot velocity evolution
    plt.subplot(2, 1, 2)
    for model, data in results.items():
        plt.plot(
            data["time_Gyr"],
            data["velocity_kms"],
            color=styles[model]["color"],
            linestyle=styles[model]["linestyle"],
            linewidth=2.5,
            label=styles[model]["label"]
        )
    
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.7)  # Zero velocity line
    plt.title("Radial Velocity Evolution", fontsize=16)
    plt.xlabel("Time (Gyr)", fontsize=14)
    plt.ylabel("Velocity (km/s)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("collapse_comparison.png", dpi=300)
    plt.show()

# Execute the simulation
print("Starting spherical collapse simulations...")
results = run_simulations()
visualize_results(results)

# Additional analysis: how the pinning model affects acceleration at different densities
def analyze_acceleration_profiles():
    print("\nAnalyzing acceleration profiles at different densities...")
    
    # Set up radius range (0.1 to 100 Mpc)
    radii_Mpc = np.logspace(-1, 2, 100)  # Log scale from 0.1 to 100 Mpc
    radii_m = radii_Mpc * Mpc_to_m
    
    # Fixed mass (equivalent to a galaxy)
    galaxy_mass = 1e12 * solar_mass  # 1 trillion solar masses
    
    # Calculate densities at each radius
    densities = np.zeros_like(radii_m)
    for i, r in enumerate(radii_m):
        if r > 0:
            densities[i] = galaxy_mass / ((4/3) * np.pi * r**3)
    
    # Calculate accelerations for each model
    accelerations = {
        "standard": np.zeros_like(radii_m),
        "lambda": np.zeros_like(radii_m),
        "pinning": np.zeros_like(radii_m)
    }
    
    for i, (r, rho) in enumerate(zip(radii_m, densities)):
        # Standard Newtonian
        accelerations["standard"][i] = -G * galaxy_mass / r**2 if r > 0 else 0
        
        # ΛCDM
        a_newton = -G * galaxy_mass / r**2 if r > 0 else 0
        a_lambda = Lambda0 * r / 3
        accelerations["lambda"][i] = a_newton + a_lambda
        
        # Pinning model
        lambda_eff = lambda_effective(rho)
        a_newton = -G * galaxy_mass / r**2 if r > 0 else 0
        a_lambda = lambda_eff * r / 3
        pinning_correction = alpha * lambda_pinning(rho) * rho * r / 6
        accelerations["pinning"][i] = a_newton + a_lambda + pinning_correction
    
    # Convert to more intuitive units (m/s²)
    for model in accelerations:
        accelerations[model] = accelerations[model]
    
    # Plot acceleration profiles
    plt.figure(figsize=(10, 8))
    
    styles = {
        "standard": {"color": "blue", "linestyle": "-", "label": "Standard Newtonian"},
        "lambda": {"color": "green", "linestyle": "--", "label": "ΛCDM Model"},
        "pinning": {"color": "red", "linestyle": "-", "label": "Density-Dependent Model"}
    }
    
    for model, accels in accelerations.items():
        plt.loglog(
            radii_Mpc,
            np.abs(accels),
            color=styles[model]["color"],
            linestyle=styles[model]["linestyle"],
            linewidth=2.5,
            label=styles[model]["label"]
        )
    
    plt.title("Radial Acceleration Profiles", fontsize=16)
    plt.xlabel("Radius (Mpc)", fontsize=14)
    plt.ylabel("Acceleration Magnitude (m/s²)", fontsize=14)
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("acceleration_profiles.png", dpi=300)
    plt.show()
    
    # Calculate circular velocities (rotation curves)
    velocities = {
        model: np.sqrt(np.abs(accels * radii_m)) / 1000  # Convert to km/s
        for model, accels in accelerations.items()
    }
    
    # Plot rotation curves
    plt.figure(figsize=(10, 8))
    
    for model, vels in velocities.items():
        plt.plot(
            radii_Mpc,
            vels,
            color=styles[model]["color"],
            linestyle=styles[model]["linestyle"],
            linewidth=2.5,
            label=styles[model]["label"]
        )
    
    plt.title("Predicted Rotation Curves", fontsize=16)
    plt.xlabel("Radius (Mpc)", fontsize=14)
    plt.ylabel("Circular Velocity (km/s)", fontsize=14)
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("rotation_curves.png", dpi=300)
    plt.show()

# Run the acceleration analysis
analyze_acceleration_profiles()