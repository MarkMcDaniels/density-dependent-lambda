import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants in astrophysical units
G_Ast = 4.30091e-6  # kpc (km/s)^2 M_⊙^-1
a0_mond_ast = 3.7  # (km/s)^2/kpc

# Model parameters
k_teves = 0.03  # TeVeS parameter
alpha_pinning = 0.05  # M_⊙^-1 kpc^3 - Pinning model parameter

# Function to load SPARC data
def load_sparc_data(filepath='data/sparc_visual_1.csv'):
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded SPARC data with {len(df)} points")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check if 'data/sparc_visual_1.csv' exists")
        return None

# MOND model - simple interpolation
def mond_simple_velocity(r, v_bary):
    """Calculate total velocity prediction using MOND with simple interpolation"""
    a_n = v_bary**2 / r  # Newtonian acceleration
    x = a_n / a0_mond_ast
    mu = x / (1 + x)  # Simple interpolation function
    v_total = np.sqrt(v_bary**2 / mu)
    return v_total

# MOND model - standard interpolation
def mond_standard_velocity(r, v_bary):
    """Calculate total velocity prediction using MOND with standard interpolation"""
    a_n = v_bary**2 / r  # Newtonian acceleration
    x = a_n / a0_mond_ast
    mu = x / np.sqrt(1 + x**2)  # Standard interpolation function
    v_total = np.sqrt(v_bary**2 / mu)
    return v_total

# TeVeS model
def teves_velocity(r, v_bary):
    """Calculate total velocity prediction using TeVeS"""
    a_n = v_bary**2 / r  # Newtonian acceleration
    y = np.sqrt(k_teves/(4*np.pi)) * np.log(1 + 4*np.pi/k_teves * (a_n/a0_mond_ast)**2) / a_n
    a_teves = a_n * (1 + y)
    v_total = np.sqrt(a_teves * r)
    return v_total

# Calculate dark matter velocity from observed and baryonic
def calculate_vdm(v_obs, v_bary):
    """Calculate the dark matter velocity component"""
    v_dm_sq = v_obs**2 - v_bary**2
    # Ensure non-negative values
    v_dm_sq = np.maximum(0, v_dm_sq)
    return np.sqrt(v_dm_sq)

# Pinning model
def pinning_velocity(r, v_bary, v_dm, alpha=alpha_pinning):
    """Calculate total velocity prediction using the Pinning model"""
    # Calculate pinning density based on observed DM component
    v_dm_sq = v_dm**2
    r_v_dm_sq = r * v_dm_sq
    
    # Calculate gradient - handle potential instabilities with smoothing
    if len(r) >= 5:
        # Use smoothed gradient for numerical stability with enough points
        from scipy.signal import savgol_filter
        window = min(5, len(r) - (len(r) % 2) - 1)  # Must be odd and <= len(r)
        if window >= 3:
            r_v_dm_sq_smooth = savgol_filter(r_v_dm_sq, window, 2)
            d_r_v_dm_sq = np.gradient(r_v_dm_sq_smooth, r)
        else:
            d_r_v_dm_sq = np.gradient(r_v_dm_sq, r)
    else:
        # Simple gradient for few points
        d_r_v_dm_sq = np.gradient(r_v_dm_sq, r)
    
    # Calculate pinning density
    rho_pinning = (1 / (4 * np.pi * G_Ast)) * (1 / r**2) * d_r_v_dm_sq
    
    # Initialize pinning mass array
    m_pinning = np.zeros_like(r)
    
    # Integrate to get pinning mass at each radius
    for i in range(1, len(r)):
        integrand = 4 * np.pi * r[:i]**2 * rho_pinning[:i]
        m_pinning[i] = np.trapz(integrand, r[:i])
    
    # Calculate pinning velocity component
    v_pinning_sq = G_Ast * m_pinning / r
    v_pinning = np.sqrt(np.maximum(0, v_pinning_sq))
    
    # Total velocity (baryonic + pinning)
    v_total = np.sqrt(v_bary**2 + v_pinning**2)
    
    return v_total

def plot_galaxy_comparison(galaxy_id, sparc_df):
    """Create a two-panel plot with full curve and outer region focus"""
    # Filter data for the specified galaxy
    gal_data = sparc_df[sparc_df['ID'] == galaxy_id].copy()
    
    # Skip if no data
    if len(gal_data) == 0:
        print(f"No data found for galaxy {galaxy_id}")
        return None
    
    # Filter out central regions (r < 1 kpc)
    gal_data = gal_data[gal_data['R'] >= 1.0].copy()
    
    # Skip if too few data points
    if len(gal_data) < 5:
        print(f"Not enough data points for galaxy {galaxy_id}")
        return None
    
    # Sort by radius
    gal_data = gal_data.sort_values('R')
    
    # Get data
    r = gal_data['R'].values
    v_obs = gal_data['Vobs'].values
    v_gas = gal_data['Vgas'].values
    v_disk = gal_data['Vdisk'].values
    v_bulge = gal_data['Vbul'].values
    v_bary = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
    
    # Calculate observed DM component
    v_dm = calculate_vdm(v_obs, v_bary)
    
    # Calculate model predictions
    v_mond_simple = mond_simple_velocity(r, v_bary)
    v_mond_std = mond_standard_velocity(r, v_bary)
    v_teves = teves_velocity(r, v_bary)
    v_pinning = pinning_velocity(r, v_bary, v_dm)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Full rotation curve
    ax1.errorbar(r, v_obs, yerr=gal_data['e_Vobs'].values, 
                 fmt='o', label='Observed', color='black', markersize=6)
    ax1.plot(r, v_bary, '--', label='Baryonic Only', color='gray', linewidth=2)
    ax1.plot(r, v_mond_simple, '-', label='MOND (simple)', color='blue', linewidth=2)
    ax1.plot(r, v_mond_std, '-', label='MOND (standard)', color='cyan', linewidth=2)
    ax1.plot(r, v_teves, '-', label='TeVeS', color='green', linewidth=2)
    ax1.plot(r, v_pinning, '-', label='Pinning Model', color='red', linewidth=2)
    
    ax1.set_xlabel('Radius (kpc)', fontsize=12)
    ax1.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax1.set_title('Full Rotation Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: Focus on outer regions
    # Determine cutoff (e.g., last 1/3 of radial points)
    outer_start_idx = int(2 * len(r) / 3)
    
    # If there are very few points, just use the outer half
    if outer_start_idx < 3:
        outer_start_idx = max(int(len(r) / 2), 1)
    
    # Outer region data
    r_outer = r[outer_start_idx:]
    v_obs_outer = v_obs[outer_start_idx:]
    v_bary_outer = v_bary[outer_start_idx:]
    v_mond_simple_outer = v_mond_simple[outer_start_idx:]
    v_mond_std_outer = v_mond_std[outer_start_idx:]
    v_teves_outer = v_teves[outer_start_idx:]
    v_pinning_outer = v_pinning[outer_start_idx:]
    v_obs_err_outer = gal_data['e_Vobs'].values[outer_start_idx:]
    
    ax2.errorbar(r_outer, v_obs_outer, yerr=v_obs_err_outer, 
                 fmt='o', label='Observed', color='black', markersize=8)
    ax2.plot(r_outer, v_bary_outer, '--', label='Baryonic Only', color='gray', linewidth=2)
    ax2.plot(r_outer, v_mond_simple_outer, '-', label='MOND (simple)', color='blue', linewidth=2)
    ax2.plot(r_outer, v_mond_std_outer, '-', label='MOND (standard)', color='cyan', linewidth=2)
    ax2.plot(r_outer, v_teves_outer, '-', label='TeVeS', color='green', linewidth=2)
    ax2.plot(r_outer, v_pinning_outer, '-', label='Pinning Model', color='red', linewidth=2)
    
    ax2.set_xlabel('Radius (kpc)', fontsize=12)
    ax2.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax2.set_title('Outer Region Focus', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Calculate RMSE for full curve
    rmse_mond_simple = np.sqrt(np.mean((v_obs - v_mond_simple)**2))
    rmse_mond_std = np.sqrt(np.mean((v_obs - v_mond_std)**2))
    rmse_teves = np.sqrt(np.mean((v_obs - v_teves)**2))
    rmse_pinning = np.sqrt(np.mean((v_obs - v_pinning)**2))
    
    # Calculate RMSE for outer regions
    rmse_mond_simple_outer = np.sqrt(np.mean((v_obs_outer - v_mond_simple_outer)**2))
    rmse_mond_std_outer = np.sqrt(np.mean((v_obs_outer - v_mond_std_outer)**2))
    rmse_teves_outer = np.sqrt(np.mean((v_obs_outer - v_teves_outer)**2))
    rmse_pinning_outer = np.sqrt(np.mean((v_obs_outer - v_pinning_outer)**2))
    
    # Calculate improvement percentages
    improve_over_mond_simple = (rmse_mond_simple - rmse_pinning) / rmse_mond_simple * 100
    improve_over_mond_std = (rmse_mond_std - rmse_pinning) / rmse_mond_std * 100
    improve_over_teves = (rmse_teves - rmse_pinning) / rmse_teves * 100
    
    # Calculate improvement percentages for outer regions
    improve_over_mond_simple_outer = (rmse_mond_simple_outer - rmse_pinning_outer) / rmse_mond_simple_outer * 100
    improve_over_mond_std_outer = (rmse_mond_std_outer - rmse_pinning_outer) / rmse_mond_std_outer * 100
    improve_over_teves_outer = (rmse_teves_outer - rmse_pinning_outer) / rmse_teves_outer * 100
    
    # Add metrics text for full curve
    full_metrics = (
        f"Full Curve RMSE (km/s):\n"
        f"MOND Simple: {rmse_mond_simple:.2f}\n"
        f"MOND Std: {rmse_mond_std:.2f}\n"
        f"TeVeS: {rmse_teves:.2f}\n"
        f"Pinning: {rmse_pinning:.2f}\n\n"
        f"Pinning Improvement:\n"
        f"vs MOND Simple: {improve_over_mond_simple:.1f}%\n"
        f"vs MOND Std: {improve_over_mond_std:.1f}%\n"
        f"vs TeVeS: {improve_over_teves:.1f}%"
    )
    
    # Add metrics text for outer regions
    outer_metrics = (
        f"Outer Region RMSE (km/s):\n"
        f"MOND Simple: {rmse_mond_simple_outer:.2f}\n"
        f"MOND Std: {rmse_mond_std_outer:.2f}\n"
        f"TeVeS: {rmse_teves_outer:.2f}\n"
        f"Pinning: {rmse_pinning_outer:.2f}\n\n"
        f"Pinning Improvement:\n"
        f"vs MOND Simple: {improve_over_mond_simple_outer:.1f}%\n"
        f"vs MOND Std: {improve_over_mond_std_outer:.1f}%\n"
        f"vs TeVeS: {improve_over_teves_outer:.1f}%"
    )
    
    # Add text boxes with metrics
    ax1.text(0.05, 0.05, full_metrics, transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8), verticalalignment='bottom')
    
    ax2.text(0.05, 0.05, outer_metrics, transform=ax2.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8), verticalalignment='bottom')
    
    plt.suptitle(f'Galaxy {galaxy_id} - Rotation Curve Model Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_dir = 'rotation_curve_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{galaxy_id}_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"Comparison plot saved for galaxy {galaxy_id}")
    return fig

def main():
    # Load SPARC data
    sparc_df = load_sparc_data()
    
    if sparc_df is None:
        return
    
    # Get list of galaxies
    galaxies = sparc_df['ID'].unique()
    
    # Create output directory
    output_dir = 'rotation_curve_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select specific galaxies of interest
    target_galaxies = ['D631-7']  # Start with the galaxy from your example
    
    # Try to find a few more galaxies with enough data points
    for gal_id in galaxies:
        if gal_id in target_galaxies:
            continue
            
        gal_data = sparc_df[sparc_df['ID'] == gal_id]
        gal_data = gal_data[gal_data['R'] >= 1.0]  # Filter out central regions
        if len(gal_data) >= 12:  # Look for galaxies with many data points
            target_galaxies.append(gal_id)
            if len(target_galaxies) >= 6:  # Limit to 6 galaxies total
                break
    
    print(f"Creating rotation curve plots for {len(target_galaxies)} galaxies...")
    
    # Plot rotation curves for target galaxies
    for gal_id in target_galaxies:
        try:
            plot_galaxy_comparison(gal_id, sparc_df)
        except Exception as e:
            print(f"Error plotting galaxy {gal_id}: {e}")
    
    print(f"All plots saved to {output_dir}/")

if __name__ == "__main__":
    main()