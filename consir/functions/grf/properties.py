import numpy as np
import consir
from consir.functions.grf import generator 

def find_baseline(curve, max_delta=2,eps=1e-3):
    """
    Find the baseline level of a curve by identifying the last negative to positive transition 
    on the left of the peak and the first such transition on the right.

    Parameters:
    curve (numpy.ndarray): 1D array representing the curve from which to find the baseline.

    Returns:
    float: The average value of the baseline points found on the left and right of the peak.
    """

    if np.std(curve) < eps:
        return np.mean(curve)
    
    peak_index = np.argmax(curve)  # Find the index of the peak
    assert abs(peak_index - len(curve)//2) < max_delta, "Peak must be near center"

    left_slope = np.diff(curve[:peak_index+1])  # Calculate the difference to the left of the peak
    right_slope = np.diff(curve[peak_index:])  # Calculate the difference to the right of the peak

    # Find the last negative to positive transition on the left
    for i in range(len(left_slope) - 2, -1, -1):
        if left_slope[i] < 0 and left_slope[i + 1] >= 0:
            left_baseline = i
            break
    else:
        left_baseline = 0  # Default to the start if no change found

    # Find the first negative to positive transition on the right
    for i in range(1, len(right_slope)):
        if right_slope[i - 1] < 0 and right_slope[i] >= 0:
            right_baseline = peak_index + i
            break
    else:
        right_baseline = len(curve) - 1  # Default to the end if no change found

    return (curve[left_baseline] + curve[right_baseline])/2.0

def compute_autocorrelation(image):
    """
    Compute the autocorrelation of a 2D image using Fourier transforms, and determine the correlation length 
    from the central slice of the autocorrelation matrix.

    Parameters:
    image (numpy.ndarray): 2D array representing the image to be analyzed.

    Returns:
    float or None: The correlation length of the image if a valid baseline is found, otherwise None.
    """
    assert len(image.shape) == 2, "Image must be a 2D array."
    assert image.shape[0] == image.shape[1], "Image must be square."

    x = np.linspace(0, 1, image.shape[0])
    ft_img = np.fft.fft2(image)
    auto_correlation = np.fft.fftshift(np.fft.ifft2(ft_img * ft_img.conj()).real)
    mid_point = image.shape[0] // 2
    mean_central_slice = 0.5 * (auto_correlation[:, mid_point] + auto_correlation[mid_point, :])
    base_level = find_baseline(mean_central_slice)

    if base_level < np.argmax(mean_central_slice):
        mean_central_slice -= base_level
        mean_central_slice /= np.max(mean_central_slice)
        sel = mean_central_slice > 1 / np.exp(1)
        these_x = x[sel]
        correlation_length = (np.max(these_x) - np.min(these_x)) / 2.0
        return correlation_length
    else:
        return None

