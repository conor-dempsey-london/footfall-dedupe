# %%
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd


def create_resident_model(
    observed_residents, 
    unique_residents, 
    area_sizes, 
    area_ids,
):
    """
    Create Bayesian hierarchical model for resident count relationship.
    
    Parameters:
    -----------
    observed_residents : array-like
        Number of observed residents for each measurement
    unique_residents : array-like  
        Number of unique residents for each measurement
    area_sizes : array-like
        Size of each area (length = n_areas)
    area_ids : array-like
        Area identifier for each measurement (0-indexed)
    
    Returns:
    --------
    pm.Model
        PyMC model object
    """
    n_areas = len(area_sizes)
    n_obs = len(observed_residents)

    with pm.Model() as model:
        # Hyperpriors for the slope distribution
        alpha = pm.Normal("alpha", mu=0, sigma=1)  # Intercept for slope-size relationship
        gamma = pm.Normal("gamma", mu=0, sigma=1)  # Effect of area size on slope
        
        # Hyperpriors for variance components
        tau = pm.HalfNormal("tau", sigma=0.5)  # SD of slopes around the regression line
        kappa = pm.HalfNormal("kappa", sigma=0.5)  # SD of observation noise SDs
        
        # Area-specific slopes - hierarchical prior
        # The mean slope for each area depends on its size
        beta_mean = alpha + gamma * area_sizes
        beta_i = pm.Normal("beta_i", mu=beta_mean, sigma=tau, shape=n_areas)
        
        # Area-specific observation noise
        sigma_i = pm.HalfNormal("sigma_i", sigma=kappa, shape=n_areas)
        
        # Expected value for each observation
        # This is the key line: beta_i[area_ids] selects the appropriate slope for each observation
        mu_obs = beta_i[area_ids] * observed_residents
        
        # Likelihood
        unique_residents_obs = pm.Normal(
            "unique_residents_obs", 
            mu=mu_obs, 
            sigma=sigma_i[area_ids], 
            observed=unique_residents
        )
    
    return model

def generate_sample_resident_data(
    n_areas=20, 
    obs_per_area=50,
    # True parameters
    true_alpha = 1.0,  # Base slope
    true_gamma = 0.3,  # Effect of area size on slope
    true_tau = 0.2,  # Variation in slopes
    true_kappa = 0.1,  # Variation in noise SDs
    base_noise_sd = 0.3,
):
    """
    Generate synthetic data matching your description.
    """
    np.random.seed(42)
    
    # Generate area sizes (e.g., square kilometers)
    area_sizes = np.random.lognormal(2, 0.5, n_areas)
    
    # Generate area-specific slopes based on size
    beta_mean = true_alpha + true_gamma * area_sizes
    beta_i_true = np.random.normal(beta_mean, true_tau, n_areas)
    
    # Generate area-specific noise SDs
    sigma_i_true = np.abs(np.random.normal(base_noise_sd, true_kappa, n_areas))
    
    # Generate observations
    observed_residents = []
    unique_residents = []
    area_ids = []
    
    for i in range(n_areas):
        # Generate observed residents for this area
        obs_area = np.random.normal(100, 20, obs_per_area)
        
        # Generate unique residents based on the linear relationship
        unique_area = beta_i_true[i] * obs_area + np.random.normal(0, sigma_i_true[i], obs_per_area)
        
        observed_residents.extend(obs_area)
        unique_residents.extend(unique_area)
        area_ids.extend([i] * obs_per_area)
    
    return (
        np.array(observed_residents),
        np.array(unique_residents), 
        area_sizes,
        np.array(area_ids),
        {"alpha": true_alpha, "gamma": true_gamma, "tau": true_tau, "kappa": true_kappa, "beta_i": beta_i_true, "sigma_i": sigma_i_true}
    )


def create_model_and_sample(
    observed_residents, 
    unique_residents, 
    area_sizes, 
    area_ids, 
    true_params,
):
    """
    Complete analysis pipeline.
    """
    print(f"Data: {len(area_ids)} observations across {len(area_sizes)} areas")
    print(f"Area sizes range: {area_sizes.min():.1f} to {area_sizes.max():.1f}")
    
    # Create and sample from model
    model = create_resident_model(observed_residents, unique_residents, area_sizes, area_ids)
    
    with model:
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)
    
    return trace, model


def plot_results_summary(
    trace,
    area_sizes, 
    true_params, 
):

    # Analyze results
    print("\n=== POSTERIOR SUMMARY ===")
    print(az.summary(trace, var_names=["alpha", "gamma", "tau", "kappa"]))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Relationship between area size and slope
    ax = axes[0, 0]
    beta_i_est = trace.posterior.beta_i.mean(dim=("chain", "draw")).values
    ax.scatter(area_sizes, beta_i_est, alpha=0.7, label='Estimated slopes')
    if true_params:
        ax.scatter(area_sizes, true_params["beta_i"], alpha=0.7, label='True slopes', color='red')
    
    # Plot regression line
    area_sizes_sorted = np.sort(area_sizes)
    alpha_est = trace.posterior.alpha.mean().item()
    gamma_est = trace.posterior.gamma.mean().item()

    regression_line = alpha_est + gamma_est * area_sizes_sorted
    ax.plot(area_sizes_sorted, regression_line, 'k--', label='Estimated relationship')
    
    ax.set_xlabel('Area Size')
    ax.set_ylabel('Slope (beta_i)')
    ax.set_title('Relationship between Area Size and Slope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Posterior distributions of hyperparameters
    az.plot_posterior(trace, var_names=["alpha"], ax=axes[0, 1])
    if true_params:
        axes[0, 1].axvline(true_params["alpha"], color='red', linestyle='--')
    
    az.plot_posterior(trace, var_names=["gamma"], ax=axes[1, 0])
    if true_params:
        axes[1, 0].axvline(true_params["gamma"], color='red', linestyle='--')
    
    az.plot_posterior(trace, var_names=["tau"], ax=axes[1, 1])
    if true_params:
        axes[1, 1].axvline(true_params["tau"], color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()


def plot_area_fits(trace, observed_residents, unique_residents, area_ids, area_sizes, area_indices):
    """
    Plot data and fits for specific areas.
    """
    n_areas = len(area_indices)
    fig, axes = plt.subplots(2, (n_areas + 1) // 2, figsize=(15, 10), 
    sharex=True, sharey=True)
    axes = axes.flatten()
    
    beta_i_est = trace.posterior.beta_i.mean(dim=("chain", "draw")).values
    
    for idx, area_idx in enumerate(area_indices):
        # Get data for this area
        area_mask = area_ids == area_idx
        obs_area = observed_residents[area_mask]
        unique_area = unique_residents[area_mask]
        
        # Get estimated slope
        beta_est = beta_i_est[area_idx]
        
        # Create regression line
        obs_range = np.linspace(obs_area.min(), obs_area.max(), 100)
        regression_line = beta_est * obs_range
        
        ax = axes[idx]
        ax.scatter(obs_area, unique_area, alpha=0.6, label=f'Area {area_idx} data')
        ax.plot(obs_range, regression_line, 'r-', linewidth=2, 
                label=f'Slope: {beta_est:.2f}\nSize: {area_sizes[area_idx]:.1f}')
        
        ax.set_xlabel('Observed Residents')
        ax.set_ylabel('Unique Residents')
        ax.set_title(f'Area {area_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Resident Count Relationships by Area', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    # %%
    #  1. generate synthetic data
    observed_residents, unique_residents, area_sizes, area_ids, true_params = generate_sample_resident_data(
        n_areas=20, 
        obs_per_area=50,
        # True parameters
        true_alpha = 1.0,  # Base slope
        true_gamma = 0.3,  # Effect of area size on slope
        true_tau = 0.4,  # Variation in slopes
        true_kappa = 0.1,  # Variation in noise SDs
        base_noise_sd = 50,
    )

    # %%
    # 2. create model and draw samples 
    trace, model = create_model_and_sample(
        observed_residents, 
        unique_residents, 
        area_sizes, 
        area_ids, 
        true_params,
    )

    # %%
    # 3. Plot results summary
    plot_results_summary(
        trace,
        area_sizes, 
        true_params, 
    )

    # %%
    # 4. Check area-specific fits
    plot_area_fits(trace, observed_residents, unique_residents, area_ids, area_sizes, [0, 1, 2, 3])

# %%
