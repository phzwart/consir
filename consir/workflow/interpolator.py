from typing import Tuple, Optional, Literal
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator

def interpolate_to_grid(x: np.ndarray,
                       y: np.ndarray,
                       signals: np.ndarray,
                       grid_shape: Tuple[int, int],
                       method: Literal['nearest', 'linear', 'cubic'] = 'linear',
                       extrapolate: bool = False) -> np.ndarray:
    """
    Interpolate scattered point measurements to a regular grid.
    
    Args:
        x (np.ndarray): x-coordinates of measurements, shape (N,)
        y (np.ndarray): y-coordinates of measurements, shape (N,)
        signals (np.ndarray): Signal values at each point, shape (N, C) where C is number of channels
        grid_shape (Tuple[int, int]): Desired output shape (Y, X)
        method (str): Interpolation method ('nearest', 'linear', or 'cubic')
        extrapolate (bool): Whether to extrapolate values outside the convex hull of input points
        
    Returns:
        np.ndarray: Interpolated values on regular grid, shape (C, Y, X)
    """
    # Input validation
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.shape[0] != signals.shape[0]:
        raise ValueError("Coordinate arrays must have same length as first dimension of signals")
    if method not in ['nearest', 'linear', 'cubic']:
        raise ValueError("Method must be 'nearest', 'linear', or 'cubic'")
    
    # Create regular grid
    y_grid = np.linspace(y.min(), y.max(), grid_shape[0])
    x_grid = np.linspace(x.min(), x.max(), grid_shape[1])
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Prepare output array
    n_channels = signals.shape[1] if signals.ndim > 1 else 1
    output = np.zeros((n_channels,) + grid_shape)
    
    # Reshape signals if needed
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    
    # Set fill value based on extrapolation preference
    fill_value = np.nan if not extrapolate else None
    
    # Interpolate each channel
    points = np.column_stack((x, y))
    for i in range(n_channels):
        values = signals[:, i]
        
        # Remove any NaN values before interpolation
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            valid_points = points[valid_mask]
            valid_values = values[valid_mask]
        else:
            valid_points = points
            valid_values = values
            
        # Skip if no valid points
        if len(valid_values) == 0:
            output[i] = np.nan
            continue
            
        interpolated = griddata(valid_points, valid_values, (xx, yy), 
                              method=method, 
                              fill_value=fill_value)
        
        if extrapolate and np.any(np.isnan(interpolated)):
            # Fill NaN values using nearest neighbor interpolation
            mask = np.isnan(interpolated)
            interpolated[mask] = griddata(
                valid_points, valid_values, (xx[mask], yy[mask]),
                method='nearest'
            )
            
        output[i] = interpolated
    
    return output

def create_interpolator(x: np.ndarray,
                       y: np.ndarray,
                       signals: np.ndarray,
                       method: Literal['nearest', 'linear', 'cubic'] = 'linear') -> RegularGridInterpolator:
    """
    Create an interpolator function that can be used multiple times.
    
    Args:
        x (np.ndarray): x-coordinates of measurements, shape (N,)
        y (np.ndarray): y-coordinates of measurements, shape (N,)
        signals (np.ndarray): Signal values at each point, shape (N, C)
        method (str): Interpolation method ('nearest', 'linear', or 'cubic')
        
    Returns:
        RegularGridInterpolator: Interpolator function
    """
    # Create regular grid from input points
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    if len(x_unique) * len(y_unique) != len(x):
        raise ValueError("Input points must be on a regular grid")
    
    # Reshape signals to grid
    grid_shape = (len(y_unique), len(x_unique))
    grid_signals = signals.reshape(grid_shape + (-1,))
    
    # Create interpolator
    return RegularGridInterpolator(
        (y_unique, x_unique),
        grid_signals,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
