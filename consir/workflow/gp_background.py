import torch
import gpytorch
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from typing import Tuple, Union, Optional
from scipy.signal import savgol_coeffs, savgol_filter

def savitzky_golay_nd(data: np.ndarray, 
                      wavenumbers: np.ndarray,
                      wavenumber_window: float = 0.1, 
                      poly_order: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter using scipy implementation.
    
    Args:
        data (np.ndarray): Input data array (1D or 2D)
        wavenumbers (np.ndarray): Wavenumber values
        wavenumber_window (float): Window size in wavenumber units (default: 0.1)
        poly_order (int): Order of polynomial to fit (default: 3)
    
    Returns:
        np.ndarray: Smoothed data
    """
    # Calculate window size in points based on wavenumber spacing
    wavenumber_spacing = np.median(np.diff(wavenumbers))
    window_points = int(wavenumber_window / wavenumber_spacing)
    
    # Ensure odd window size
    if window_points % 2 == 0:
        window_points += 1
    
    # Validate window size
    if window_points < 3:
        raise ValueError("Window size must be at least 3 points.")
    if window_points < poly_order + 2:
        raise ValueError(f"Window size ({window_points}) must be greater than polynomial order + 2 ({poly_order + 2})")
        
    # Handle different input dimensions
    if data.ndim == 1:
        return savgol_filter(data, window_points, poly_order, mode='mirror')
    elif data.ndim == 2:
        return np.array([
            savgol_filter(spectrum, window_points, poly_order, mode='mirror')
            for spectrum in data
        ])
    else:
        raise ValueError("Input data must be 1D or 2D array")


class GPBackgroundModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='rbf', lengthscale=500.0, device='cpu'):
        """
        Initialize GP background model.
        
        Args:
            train_x: Training input points
            train_y: Training target values
            likelihood: GPyTorch likelihood object
            kernel: Kernel type ('rbf' or 'matern')
            lengthscale: Initial lengthscale for kernel
            device: Computation device
        """
        super(GPBackgroundModel, self).__init__(train_x.to(device), train_y.to(device), likelihood.to(device))
        
        # Add validation
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise ValueError("train_x and train_y must be torch tensors")
        if kernel not in ['rbf', 'matern']:
            raise ValueError("kernel must be 'rbf' or 'matern'")
            
        # Compute least squares fit for initialization
        train_x_np = train_x.cpu().numpy().reshape(-1, 1)
        train_y_np = train_y.cpu().numpy().reshape(-1, 1)

        reg = LinearRegression().fit(train_x_np, train_y_np)
        slope = float(reg.coef_[0][0])
        intercept = float(reg.intercept_[0])

        # Initialize the mean function with the fitted values
        self.mean_module = gpytorch.means.LinearMean(input_size=1).to(device)
        self.mean_module.initialize(
            bias=torch.tensor(intercept, dtype=torch.float32, device=device),
            weights=torch.tensor([slope], dtype=torch.float32, device=device)
        )

        # Define covariance function based on kernel choice
        if kernel == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device)
            self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        elif kernel == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            ).to(device)
            self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        else:
            raise ValueError("Unsupported kernel. Use 'rbf' or 'matern'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def compute_sigmas(spectrum, tie_points, window_length, min_noise=1e-3):
    """Compute standard deviation of the spectrum around each tie point within a window."""
    sigmas = []
    for tie in tie_points:
        idx = np.abs(spectrum[:, 0] - tie).argmin()
        start = max(0, idx - window_length // 2)
        end = min(len(spectrum), idx + window_length // 2)
        sigma_value = np.std(spectrum[start:end, 1])
        # Ensure minimum noise level to avoid numerical instabilities
        sigmas.append(max(sigma_value, min_noise))
    return torch.tensor(sigmas, dtype=torch.float32)

def subtract_background(wavenumbers, spectra, tie_wavenumbers, lengthscale=10.0, kernel='rbf', 
                       window_length=50, device='cpu', scale_data=True, min_noise=1e-3, n_epochs=150):
    """
    Fits a slow-varying background using Gaussian Process Regression with specified parameters.

    Args:
        wavenumbers (array): Array of wavenumber values.
        spectra (array): Array of spectral intensity values.
        tie_wavenumbers (array): Specific wavenumbers through which the function must pass.
        lengthscale (float): Length scale for the kernel.
        kernel (str): Choice of kernel ('rbf' or 'matern').
        window_length (int): Window length for computing sigma values.
        device (str): Computation device ('cpu', 'cuda', 'mps').
        scale_data (bool): Whether to standardize data to mean=0 and variance=1.
        min_noise (float): Minimum noise level to ensure numerical stability

    Returns:
        Tuple[np.ndarray, np.ndarray]: (corrected_spectra, fitted_background)
    """
    # Fix device selection logic
    device = torch.device(device if torch.cuda.is_available() or 
                         (hasattr(torch, 'backends') and 
                          hasattr(torch.backends, 'mps') and 
                          torch.backends.mps.is_available())
                         else 'cpu')

    # Input validation
    if not isinstance(wavenumbers, np.ndarray) or not isinstance(spectra, np.ndarray):
        raise ValueError("wavenumbers and spectra must be numpy arrays")
    if len(wavenumbers) != len(spectra):
        raise ValueError("wavenumbers and spectra must have same length")
    if not isinstance(tie_wavenumbers, np.ndarray):
        raise ValueError("tie_wavenumbers must be a numpy array")

    # Convert inputs to torch tensors and move to selected device
    x = torch.tensor(wavenumbers, dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.tensor(spectra, dtype=torch.float32, device=device)
    tie_x = torch.tensor(tie_wavenumbers, dtype=torch.float32, device=device).unsqueeze(1)

    # Compute initial sigmas before scaling
    raw_sigmas = compute_sigmas(
        np.column_stack((wavenumbers, spectra)), 
        tie_wavenumbers, 
        window_length, 
        min_noise=min_noise
    ).to(device)

    # Scale the data if requested
    if scale_data:
        mean_y = y.mean()
        std_y = y.std()
        y = (y - mean_y) / std_y
        tie_y = (torch.tensor([spectra[np.abs(wavenumbers - tie).argmin()] 
                             for tie in tie_wavenumbers], 
                dtype=torch.float32, device=device) - mean_y) / std_y
        # Scale sigmas by the same factor
        sigmas = raw_sigmas / std_y
    else:
        tie_y = torch.tensor([spectra[np.abs(wavenumbers - tie).argmin()] 
                            for tie in tie_wavenumbers], 
                            dtype=torch.float32, device=device)
        mean_y, std_y = 0.0, 1.0
        sigmas = raw_sigmas

    # Define likelihood with scaled noise levels
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=sigmas**2,
        learn_additional_noise=True
    ).to(device)

    # Create and train model
    model = GPBackgroundModel(tie_x, tie_y, likelihood, kernel=kernel, 
                            lengthscale=lengthscale, device=device)

    # Training process
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.mean_module.parameters(), 'lr': 0.01},
        {'params': model.covar_module.parameters(), 'lr': 0.01}
    ])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training with convergence criteria
    prev_loss = float('inf')
    patience = 3
    min_delta = 1e-4
    patience_counter = 0
    
    for epoch in range(n_epochs):  # Increased max epochs
        optimizer.zero_grad()
        output = model(tie_x)
        loss = -mll(output, tie_y)
        loss.backward()
        optimizer.step()
        
        # Check convergence
        current_loss = loss.item()
        if abs(prev_loss - current_loss) < min_delta:
            patience_counter += 1
            if patience_counter >= patience:
                break
        else:
            patience_counter = 0
        prev_loss = current_loss

    # Evaluate the model
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_background = likelihood(model(x)).mean

    # Undo the scaling to return to original scale
    if scale_data:
        pred_background = pred_background * std_y + mean_y
        corrected_spectra = (y * std_y + mean_y) - pred_background
    else:
        corrected_spectra = y - pred_background

    return corrected_spectra.cpu().numpy(), pred_background.cpu().numpy()

def compute_rubberband_background(wavenumbers: np.ndarray, 
                                spectra: np.ndarray, 
                                n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rubberband background correction by finding the convex hull.
    
    Args:
        wavenumbers (np.ndarray): Array of wavenumber values
        spectra (np.ndarray): Array of spectral intensity values
        n_points (Optional[int]): Number of points for interpolation. If None, uses original wavenumbers
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (corrected_spectra, background)
    """
    # Input validation
    if not isinstance(wavenumbers, np.ndarray) or not isinstance(spectra, np.ndarray):
        raise ValueError("wavenumbers and spectra must be numpy arrays")
    if len(wavenumbers) != len(spectra):
        raise ValueError("wavenumbers and spectra must have same length")
    
    # Create points for ConvexHull
    points = np.column_stack([wavenumbers, spectra])
    
    # Find convex hull
    hull = ConvexHull(points)
    
    # Get vertices sorted by x-coordinate
    vertices = hull.vertices
    vertices = vertices[np.argsort(points[vertices, 0])]
    
    # Always include first and last points
    lower_hull_indices = [vertices[0]]
    
    # Find lower hull points
    for i in range(1, len(vertices)-1):
        # Get current point and its neighbors
        prev_point = points[lower_hull_indices[-1]]
        curr_point = points[vertices[i]]
        next_point = points[vertices[i+1]]
        
        # Calculate slopes
        slope_prev = (curr_point[1] - prev_point[1]) / (curr_point[0] - prev_point[0])
        slope_next = (next_point[1] - curr_point[1]) / (next_point[0] - curr_point[0])
        
        # Add point if it creates a concave up shape
        if slope_next > slope_prev:
            lower_hull_indices.append(vertices[i])
    
    # Always add last point
    lower_hull_indices.append(vertices[-1])
    
    # Get coordinates of lower hull points
    hull_x = points[lower_hull_indices, 0]
    hull_y = points[lower_hull_indices, 1]
    
    # Handle potential division by zero in interpolation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Create interpolation function with robust error handling
        f = interp1d(hull_x, hull_y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Generate background using either original or resampled points
        if n_points is not None:
            # Ensure we don't have duplicate x values
            interp_x = np.linspace(wavenumbers.min(), wavenumbers.max(), n_points)
            background = f(interp_x)
            
            # Handle any NaN values that might have been generated
            mask = np.isfinite(background)
            if not np.all(mask):
                # Interpolate only valid values
                f_clean = interp1d(
                    interp_x[mask], 
                    background[mask], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                background = f_clean(interp_x)
            
            # Interpolate back to original wavenumbers
            f_final = interp1d(
                interp_x, 
                background, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            background = f_final(wavenumbers)
        else:
            background = f(wavenumbers)
        
        # Handle any remaining NaN values
        if np.any(np.isnan(background)):
            background = np.nan_to_num(background, nan=np.nanmean(background))
    
    # Compute corrected spectra
    corrected_spectra = spectra - background
    
    return corrected_spectra, background

def subtract_rubberband_baseline(wavenumbers: np.ndarray, 
                               spectra: np.ndarray, 
                               n_points: int = 100,
                               return_background: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Wrapper function to subtract rubberband baseline from spectra.
    
    Args:
        wavenumbers (np.ndarray): Array of wavenumber values
        spectra (np.ndarray): Array of spectral intensity values
        n_points (int): Number of points for interpolation
        return_background (bool): If True, returns both corrected spectra and background
        
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            If return_background is False: corrected_spectra
            If return_background is True: (corrected_spectra, background)
    """
    # Handle 2D spectra
    if spectra.ndim == 2:
        corrected_spectra = []
        backgrounds = []
        
        for spectrum in spectra:
            corr, bg = compute_rubberband_background(wavenumbers, spectrum, n_points)
            corrected_spectra.append(corr)
            backgrounds.append(bg)
            
        corrected_spectra = np.array(corrected_spectra)
        backgrounds = np.array(backgrounds)
    else:
        corrected_spectra, backgrounds = compute_rubberband_background(wavenumbers, spectra, n_points)
    
    if return_background:
        return corrected_spectra, backgrounds
    return corrected_spectra

def process_spectrum(wavenumbers: np.ndarray,
                    spectrum: np.ndarray,
                    tie_points: np.ndarray,
                    sg_window: float = 75,
                    gp_lengthscale: float = 350.0,
                    gp_window_length: int = 50,
                    gp_epochs: int = 50,
                    device: str = 'cpu',
                    return_intermediates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Process a spectrum through a pipeline of Savitzky-Golay smoothing,
    GP background subtraction, and rubberband correction.
    
    Args:
        wavenumbers (np.ndarray): Wavenumber values
        spectrum (np.ndarray): Spectral intensity values
        tie_points (np.ndarray): Tie points for GP background
        sg_window (float): Savitzky-Golay window size
        gp_lengthscale (float): GP kernel lengthscale
        gp_window_length (int): GP window length for noise estimation
        gp_epochs (int): Number of GP training epochs
        device (str): Computation device ('cpu' or 'cuda')
        return_intermediates (bool): If True, return intermediate results
        
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, dict]]: 
            If return_intermediates=False: final corrected spectrum
            If return_intermediates=True: (final corrected spectrum, dict of intermediate results)
    """
    # Step 1: Savitzky-Golay smoothing
    smoothed = savitzky_golay_nd(spectrum, wavenumbers, sg_window)
    
    # Step 2: GP background subtraction
    gp_corrected, gp_background = subtract_background(
        wavenumbers=wavenumbers,
        spectra=smoothed,
        tie_wavenumbers=tie_points,
        lengthscale=gp_lengthscale,
        kernel='rbf',
        window_length=gp_window_length,
        device=device,
        scale_data=True,
        n_epochs=gp_epochs
    )
    
    # Step 3: Rubberband correction
    final_corrected, rubberband_background = subtract_rubberband_baseline(
        wavenumbers=wavenumbers,
        spectra=gp_corrected,
        n_points=None,
        return_background=True
    )
    
    if return_intermediates:
        intermediates = {
            'smoothed': smoothed,
            'gp_corrected': gp_corrected,
            'gp_background': gp_background,
            'rubberband_background': rubberband_background,
            'final': final_corrected
        }
        return final_corrected, intermediates
    
    return final_corrected
