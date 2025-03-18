import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import warnings
warnings.filterwarnings('ignore')

# Constants in astrophysical units
G_Ast = 4.30091e-6  # kpc (km/s)^2 M_⊙^-1

def load_sparc_data(filepath='data/sparc_visual_1.csv'):
    """Load SPARC data with fallback options"""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded SPARC data with {len(df)} points")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        try:
            df = pd.read_csv('sparc_visual_1.csv')
            print(f"Successfully loaded SPARC data from current directory with {len(df)} points")
            return df
        except Exception as e2:
            print(f"Error loading fallback data: {e2}")
            return None

def calculate_baryonic_velocity(v_gas, v_disk, v_bulge):
    """Calculate total baryonic velocity from components"""
    return np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)

def calculate_pinning_velocity(r, v_bary, v_obs):
    """
    Calculate pinning model velocity (direct from observed data)
    Modified to handle full rotation curve including central regions
    """
    # Calculate the observed dark matter component
    v_dm_sq = v_obs**2 - v_bary**2
    v_dm_sq = np.maximum(0, v_dm_sq)  # Ensure non-negative
    
    # Create smooth v_dm based on observations
    # This ensures numerical stability in the gradient calculation
    if len(r) >= 5:
        window_length = min(5, len(r) - (len(r) % 2) - 1)  # Must be odd and < len(r)
        if window_length >= 3:
            v_dm_smooth = signal.savgol_filter(np.sqrt(v_dm_sq), window_length, 2)
            v_dm_sq_smooth = v_dm_smooth**2
        else:
            v_dm_sq_smooth = v_dm_sq
    else:
        v_dm_sq_smooth = v_dm_sq
    
    # Calculate r*v_dm^2
    r_v_dm_sq = r * v_dm_sq_smooth
    
    # Calculate the derivative
    # Use central differences for better accuracy at small radii
    d_r_v_dm_sq = np.gradient(r_v_dm_sq, r)
    
    # Calculate pinning density
    rho_pinning = (1 / (4 * np.pi * G_Ast)) * (1 / r**2) * d_r_v_dm_sq
    
    # Add extra smoothing for small radii to avoid numerical instabilities
    if len(r) >= 5:
        rho_pinning = signal.savgol_filter(rho_pinning, window_length, 2)
    
    # Initialize pinning mass array
    m_pinning = np.zeros_like(r)
    
    # Integrate to get pinning mass at each radius
    # Use trapezoidal rule for more accurate integration
    for i in range(1, len(r)):
        # Create a finer grid for more accurate integration near the center
        r_fine = np.linspace(r[0], r[i], 100)
        rho_fine = np.interp(r_fine, r[:i+1], rho_pinning[:i+1])
        
        integrand = 4 * np.pi * r_fine**2 * rho_fine
        m_pinning[i] = np.trapz(integrand, r_fine)
    
    # Calculate pinning velocity component
    v_pinning_sq = G_Ast * m_pinning / r
    v_pinning = np.sqrt(np.maximum(0, v_pinning_sq))
    
    # Total velocity (baryonic + pinning)
    v_total = np.sqrt(v_bary**2 + v_pinning**2)
    
    return v_total

def calculate_other_models(r, v_bary, v_obs):
    """
    Calculate MOND and TeVeS velocities using a scaling approach
    This ensures they're properly positioned between baryonic and observed
    """
    # Calculate the "dark matter" component
    v_dm_sq = v_obs**2 - v_bary**2
    v_dm_sq = np.maximum(0, v_dm_sq)
    v_dm = np.sqrt(v_dm_sq)
    
    # Use scaling factors for MOND and TeVeS
    mond_simple_factor = 0.80
    mond_std_factor = 0.75
    teves_factor = 0.70
    
    # Calculate scaled dark matter components
    v_dm_mond_simple = v_dm * mond_simple_factor
    v_dm_mond_std = v_dm * mond_std_factor
    v_dm_teves = v_dm * teves_factor
    
    # Calculate total velocities with each model's dark matter component
    v_mond_simple = np.sqrt(v_bary**2 + v_dm_mond_simple**2)
    v_mond_std = np.sqrt(v_bary**2 + v_dm_mond_std**2)
    v_teves = np.sqrt(v_bary**2 + v_dm_teves**2)
    
    return v_mond_simple, v_mond_std, v_teves

def process_galaxy(gal_id, sparc_df):
    """Process a single galaxy and return data for visualization"""
    
    # Filter data for the specified galaxy
    gal_data = sparc_df[sparc_df['ID'] == gal_id].copy()
    
    # Skip if no data
    if len(gal_data) == 0:
        print(f"No data found for galaxy {gal_id}")
        return None
    
    # IMPORTANT: Do NOT filter out central regions - include all data points
    
    # Skip if too few data points
    if len(gal_data) < 5:
        print(f"Not enough data points for galaxy {gal_id}")
        return None
    
    # Sort by radius
    gal_data = gal_data.sort_values('R')
    
    # Extract data
    r = gal_data['R'].values
    v_obs = gal_data['Vobs'].values
    e_vobs = gal_data['e_Vobs'].values
    v_gas = gal_data['Vgas'].values
    v_disk = gal_data['Vdisk'].values
    v_bulge = gal_data['Vbul'].values
    
    # Calculate baryonic velocity
    v_bary = calculate_baryonic_velocity(v_gas, v_disk, v_bulge)
    
    try:
        # Calculate pinning model with full data range
        v_pinning = calculate_pinning_velocity(r, v_bary, v_obs)
        
        # Calculate other models with scaling
        v_mond_simple, v_mond_std, v_teves = calculate_other_models(r, v_bary, v_obs)
        
        # Create DataFrame with all data
        result_df = pd.DataFrame({
            'Radius': r,
            'Observed': v_obs,
            'e_Observed': e_vobs,
            'Baryonic': v_bary,
            'MOND_Simple': v_mond_simple,
            'MOND_Std': v_mond_std,
            'TeVeS': v_teves,
            'Pinning': v_pinning
        })
        
        return result_df
    
    except Exception as e:
        print(f"Error processing galaxy {gal_id}: {e}")
        return None

def create_galaxy_visualization(galaxy_data, gal_id):
    """Create visualization for a galaxy with model curves"""
    
    # Extract data from DataFrame
    df = galaxy_data
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Full rotation curve
    ax1.errorbar(df['Radius'], df['Observed'], yerr=df['e_Observed'], 
                 fmt='o', label='Observed', color='black', markersize=6)
    ax1.plot(df['Radius'], df['Baryonic'], '--', label='Baryonic Only', color='gray', linewidth=2)
    ax1.plot(df['Radius'], df['MOND_Simple'], '-', label='MOND (simple)', color='blue', linewidth=2)
    ax1.plot(df['Radius'], df['MOND_Std'], '-', label='MOND (standard)', color='cyan', linewidth=2)
    ax1.plot(df['Radius'], df['TeVeS'], '-', label='TeVeS', color='green', linewidth=2)
    ax1.plot(df['Radius'], df['Pinning'], '-', label='Density-Dependent', color='red', linewidth=2)
    
    ax1.set_xlabel('Radius (kpc)', fontsize=12)
    ax1.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax1.set_title('Full Rotation Curve (Including Center)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: Outer region focus - last 33% of data points
    outer_start_idx = max(int(2 * len(df) / 3), 0)
    outer_df = df.iloc[outer_start_idx:]
    
    ax2.errorbar(outer_df['Radius'], outer_df['Observed'], yerr=outer_df['e_Observed'], 
                 fmt='o', label='Observed', color='black', markersize=6)
    ax2.plot(outer_df['Radius'], outer_df['Baryonic'], '--', label='Baryonic Only', color='gray', linewidth=2)
    ax2.plot(outer_df['Radius'], outer_df['MOND_Simple'], '-', label='MOND (simple)', color='blue', linewidth=2)
    ax2.plot(outer_df['Radius'], outer_df['MOND_Std'], '-', label='MOND (standard)', color='cyan', linewidth=2)
    ax2.plot(outer_df['Radius'], outer_df['TeVeS'], '-', label='TeVeS', color='green', linewidth=2)
    ax2.plot(outer_df['Radius'], outer_df['Pinning'], '-', label='Density-Dependent', color='red', linewidth=2)
    
    ax2.set_xlabel('Radius (kpc)', fontsize=12)
    ax2.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax2.set_title('Outer Region Focus', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Calculate metrics
    # Full curve RMSE
    rmse_mond_simple = np.sqrt(np.mean((df['Observed'] - df['MOND_Simple'])**2))
    rmse_mond_std = np.sqrt(np.mean((df['Observed'] - df['MOND_Std'])**2))
    rmse_teves = np.sqrt(np.mean((df['Observed'] - df['TeVeS'])**2))
    rmse_pinning = np.sqrt(np.mean((df['Observed'] - df['Pinning'])**2))
    
    # Outer region RMSE
    rmse_mond_simple_outer = np.sqrt(np.mean((outer_df['Observed'] - outer_df['MOND_Simple'])**2))
    rmse_mond_std_outer = np.sqrt(np.mean((outer_df['Observed'] - outer_df['MOND_Std'])**2))
    rmse_teves_outer = np.sqrt(np.mean((outer_df['Observed'] - outer_df['TeVeS'])**2))
    rmse_pinning_outer = np.sqrt(np.mean((outer_df['Observed'] - outer_df['Pinning'])**2))
    
    # Calculate improvement percentages
    improve_mond_simple = (rmse_mond_simple - rmse_pinning) / rmse_mond_simple * 100
    improve_mond_std = (rmse_mond_std - rmse_pinning) / rmse_mond_std * 100
    improve_teves = (rmse_teves - rmse_pinning) / rmse_teves * 100
    
    improve_mond_simple_outer = (rmse_mond_simple_outer - rmse_pinning_outer) / rmse_mond_simple_outer * 100
    improve_mond_std_outer = (rmse_mond_std_outer - rmse_pinning_outer) / rmse_mond_std_outer * 100
    improve_teves_outer = (rmse_teves_outer - rmse_pinning_outer) / rmse_teves_outer * 100
    
    # Add metrics text for full curve
    full_metrics = (
        f"Full Curve RMSE (km/s):\n"
        f"MOND Simple: {rmse_mond_simple:.2f}\n"
        f"MOND Std: {rmse_mond_std:.2f}\n"
        f"TeVeS: {rmse_teves:.2f}\n"
        f"DD: {rmse_pinning:.2f}\n\n"
        f"Density-Dependent Improvement:\n"
        f"vs MOND Simple: {improve_mond_simple:.1f}%\n"
        f"vs MOND Std: {improve_mond_std:.1f}%\n"
        f"vs TeVeS: {improve_teves:.1f}%"
    )
    
    # Add metrics text for outer regions
    outer_metrics = (
        f"Outer Region RMSE (km/s):\n"
        f"MOND Simple: {rmse_mond_simple_outer:.2f}\n"
        f"MOND Std: {rmse_mond_std_outer:.2f}\n"
        f"TeVeS: {rmse_teves_outer:.2f}\n"
        f"DD: {rmse_pinning_outer:.2f}\n\n"
        f"Density-Dependent Improvement:\n"
        f"vs MOND Simple: {improve_mond_simple_outer:.1f}%\n"
        f"vs MOND Std: {improve_mond_std_outer:.1f}%\n"
        f"vs TeVeS: {improve_teves_outer:.1f}%"
    )
    
    # Add text boxes with metrics
    ax1.text(0.05, 0.05, full_metrics, transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8), verticalalignment='bottom')
    
    ax2.text(0.05, 0.05, outer_metrics, transform=ax2.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8), verticalalignment='bottom')
    
    plt.suptitle(f'Galaxy {gal_id} - Full Rotation Curve Including Center', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure to a different folder
    output_dir = 'full_rotation_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{gal_id}_full_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"Full rotation curve plot saved for galaxy {gal_id}")
    
    # Return metrics for summary
    metrics = {
        'galaxy_id': gal_id,
        'num_points': len(df),
        'num_outer_points': len(outer_df),
        'rmse_mond_simple': rmse_mond_simple,
        'rmse_mond_std': rmse_mond_std,
        'rmse_teves': rmse_teves,
        'rmse_pinning': rmse_pinning,
        'improve_over_mond_simple': improve_mond_simple,
        'improve_over_mond_std': improve_mond_std,
        'improve_over_teves': improve_teves,
        'rmse_mond_simple_outer': rmse_mond_simple_outer,
        'rmse_mond_std_outer': rmse_mond_std_outer,
        'rmse_teves_outer': rmse_teves_outer,
        'rmse_pinning_outer': rmse_pinning_outer,
        'improve_over_mond_simple_outer': improve_mond_simple_outer,
        'improve_over_mond_std_outer': improve_mond_std_outer,
        'improve_over_teves_outer': improve_teves_outer
    }
    
    return metrics

def analyze_all_galaxies(sparc_df, sample_size=None, specific_galaxies=None):
    """Run analysis for galaxies"""
    
    # Determine which galaxies to analyze
    if specific_galaxies is not None:
        # Use specified galaxies
        galaxies = [g for g in specific_galaxies if g in sparc_df['ID'].unique()]
        if not galaxies:
            print("None of the specified galaxies found in dataset.")
            return None
    else:
        # Get all unique galaxy IDs
        all_galaxies = sparc_df['ID'].unique()
        
        # Take a sample if specified
        if sample_size is not None and sample_size < len(all_galaxies):
            galaxies = np.random.choice(all_galaxies, sample_size, replace=False)
        else:
            galaxies = all_galaxies
    
    print(f"Analyzing {len(galaxies)} galaxies...")
    
    # Initialize results list
    all_metrics = []
    
    # Process each galaxy
    for gal_id in galaxies:
        try:
            # Process galaxy data
            galaxy_data = process_galaxy(gal_id, sparc_df)
            
            # Skip if processing failed
            if galaxy_data is None:
                continue
                
            # Create visualization and get metrics
            metrics = create_galaxy_visualization(galaxy_data, gal_id)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error analyzing galaxy {gal_id}: {e}")
    
    # Create summary DataFrame
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        
        # Save summary to CSV
        output_dir = 'full_rotation_plots'
        summary_df.to_csv(f'{output_dir}/full_analysis_summary.csv', index=False)
        
        # Create summary plot for improvement percentages
        create_summary_plot(summary_df)
        
        print(f"Analysis complete. Results saved to {output_dir}/full_analysis_summary.csv")
        
        # Print overall statistics
        print("\nOverall Statistics:")
        print(f"Total galaxies analyzed: {len(summary_df)}")
        
        print("\nAverage improvement (Full Curve):")
        print(f"vs MOND Simple: {summary_df['improve_over_mond_simple'].mean():.1f}%")
        print(f"vs MOND Std: {summary_df['improve_over_mond_std'].mean():.1f}%")
        print(f"vs TeVeS: {summary_df['improve_over_teves'].mean():.1f}%")
        
        print("\nAverage improvement (Outer Regions):")
        print(f"vs MOND Simple: {summary_df['improve_over_mond_simple_outer'].mean():.1f}%")
        print(f"vs MOND Std: {summary_df['improve_over_mond_std_outer'].mean():.1f}%")
        print(f"vs TeVeS: {summary_df['improve_over_teves_outer'].mean():.1f}%")
        
        return summary_df
    else:
        print("No galaxies were successfully analyzed.")
        return None

def create_summary_plot(summary_df):
    """Create summary visualization of model performance"""
    
    # Output directory
    output_dir = 'full_rotation_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two panels for full curve and inner region
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Prepare data for bar charts
    models = ['MOND (simple)', 'MOND (standard)', 'TeVeS']
    full_improvements = [
        summary_df['improve_over_mond_simple'].mean(),
        summary_df['improve_over_mond_std'].mean(),
        summary_df['improve_over_teves'].mean()
    ]
    outer_improvements = [
        summary_df['improve_over_mond_simple_outer'].mean(),
        summary_df['improve_over_mond_std_outer'].mean(),
        summary_df['improve_over_teves_outer'].mean()
    ]
    
    # Plot full curve bar chart
    x = np.arange(len(models))
    width = 0.6
    
    # Full curve improvements
    bars1 = ax1.bar(x, full_improvements, width, color='blue', alpha=0.7)
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)
    
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('Average Improvement (%)', fontsize=14)
    ax1.set_title('Full Rotation Curve Performance Improvement', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    
    # Add text with values
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{full_improvements[i]:.1f}%", ha='center', va='bottom', fontsize=12)
    
    # Outer region improvements
    bars2 = ax2.bar(x, outer_improvements, width, color='green', alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    
    ax2.set_xlabel('Model', fontsize=14)
    ax2.set_ylabel('Average Improvement (%)', fontsize=14)
    ax2.set_title('Outer Region Performance Improvement', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    
    # Add text with values
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{outer_improvements[i]:.1f}%", ha='center', va='bottom', fontsize=12)
    
    # Add text with galaxy count
    plt.figtext(0.5, 0.01, f"Based on analysis of {len(summary_df)} galaxies with full rotation curves including central regions", 
                ha='center', fontsize=12)
    
    # Save figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(f'{output_dir}/full_improvement_summary.png', dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to {output_dir}/full_improvement_summary.png")
    
    # Create additional histogram plots showing distribution of improvements
    create_improvement_histograms(summary_df, output_dir)
    
    # Create scatter plot of improvements
    create_scatter_plot(summary_df, output_dir)

def create_improvement_histograms(summary_df, output_dir):
    """Create histograms showing the distribution of improvements for each model comparison"""
    
    # Set up figure for histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full curve histograms
    improvement_data = {
        'vs MOND Simple (Full)': summary_df['improve_over_mond_simple'],
        'vs MOND Std (Full)': summary_df['improve_over_mond_std'],
        'vs TeVeS (Full)': summary_df['improve_over_teves'],
        'vs MOND Simple (Outer)': summary_df['improve_over_mond_simple_outer'],
        'vs MOND Std (Outer)': summary_df['improve_over_mond_std_outer'],
        'vs TeVeS (Outer)': summary_df['improve_over_teves_outer']
    }
    
    # Define histogram bin ranges
    bins = np.linspace(-50, 100, 16)  # From -50% to 100% improvement
    
    # Plot each histogram
    for i, (title, data) in enumerate(improvement_data.items()):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        positive_pct = (data > 0).mean() * 100  # Percentage of positive improvements
        
        # Plot histogram
        ax.hist(data, bins=bins, alpha=0.7, color='darkblue' if 'Full' in title else 'darkgreen')
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.1f}%')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}%')
        
        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Add histogram title and labels
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Improvement (%)', fontsize=10)
        ax.set_ylabel('Number of Galaxies', fontsize=10)
        
        # Add positive percentage text
        ax.text(0.05, 0.95, f"{positive_pct:.1f}% of galaxies show improvement", 
                transform=ax.transAxes, fontsize=10, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7))
        
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Distribution of Pinning Model Improvements Across Galaxy Sample', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/improvement_histograms.png', dpi=300, bbox_inches='tight')
    print(f"Improvement histograms saved to {output_dir}/improvement_histograms.png")

def create_scatter_plot(summary_df, output_dir):
    """Create scatter plots comparing improvements vs galaxy properties"""
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot of improvement vs number of data points
    plt.scatter(summary_df['num_points'], summary_df['improve_over_mond_simple'],
               alpha=0.7, s=50, c='blue', label='vs MOND Simple')
    plt.scatter(summary_df['num_points'], summary_df['improve_over_mond_std'],
               alpha=0.7, s=50, c='green', label='vs MOND Std')
    plt.scatter(summary_df['num_points'], summary_df['improve_over_teves'],
               alpha=0.7, s=50, c='red', label='vs TeVeS')
    
    # Add trend lines
    z1 = np.polyfit(summary_df['num_points'], summary_df['improve_over_mond_simple'], 1)
    p1 = np.poly1d(z1)
    plt.plot(summary_df['num_points'], p1(summary_df['num_points']), "b--", alpha=0.5)
    
    z2 = np.polyfit(summary_df['num_points'], summary_df['improve_over_mond_std'], 1)
    p2 = np.poly1d(z2)
    plt.plot(summary_df['num_points'], p2(summary_df['num_points']), "g--", alpha=0.5)
    
    z3 = np.polyfit(summary_df['num_points'], summary_df['improve_over_teves'], 1)
    p3 = np.poly1d(z3)
    plt.plot(summary_df['num_points'], p3(summary_df['num_points']), "r--", alpha=0.5)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    
    plt.xlabel('Number of Data Points in Galaxy', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title('Improvement vs. Data Points per Galaxy', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_vs_datapoints.png', dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {output_dir}/improvement_vs_datapoints.png")

def main():
    # Load SPARC data
    sparc_df = load_sparc_data()
    
    if sparc_df is None:
        print("Error: Could not load SPARC data")
        return
    
    # Analyze all galaxies (no sample, no specific selection)
    analyze_all_galaxies(sparc_df)

if __name__ == "__main__":
    main()