import numpy as np
from scipy import ndimage
from scipy.fft import next_fast_len
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

_INTENSITY_THRESHOLD = 1e-10
_SIGMOID_STEEPNESS = 4.0 

# ============================================================================
# INPUT VALIDATION
# ============================================================================

def _validate_2d_array(array: np.ndarray, name: str = "array") -> None:
    """Validate that input is a 2D numpy array."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.ndim}D with shape {array.shape}")


def _validate_1d_array(array: np.ndarray, name: str = "array") -> None:
    """Validate that input is a 1D numpy array."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {array.ndim}D")


def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_coordinate_match(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> None:
    """Validate that coordinate arrays match field dimensions."""
    if len(x) != field.shape[1]:
        raise ValueError(
            f"x coordinate length ({len(x)}) doesn't match field width ({field.shape[1]})"
        )
    if len(y) != field.shape[0]:
        raise ValueError(
            f"y coordinate length ({len(y)}) doesn't match field height ({field.shape[0]})"
        )


# ============================================================================
# PADDING AND CROPPING
# ============================================================================

def zero_padding_2N(array: np.ndarray) -> np.ndarray:
    """
    Pad array to double its size on all sides.
    
    This function pads the input array with zeros, doubling the size in each
    dimension. Useful for avoiding circular convolution artifacts in FFT-based
    operations.
    
    Parameters
    ----------
    array : np.ndarray
        Input 2D array
        
    Returns
    -------
    np.ndarray
        Padded array with shape (2*N, 2*M)
        
    Raises
    ------
    TypeError, ValueError
        If array is not a 2D numpy array
        
    Examples
    --------
    >>> arr = np.ones((100, 100))
    >>> padded = zero_padding_2N(arr)
    >>> padded.shape
    (200, 200)
    """
    _validate_2d_array(array, "array")
    
    N, M = array.shape
    pad_height = N // 2
    pad_width = M // 2
    
    return np.pad(
        array,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant',
        constant_values=0
    )


def clip_to_original_size(padded: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Center-crop padded array back to original size.
    
    Extracts the central region from a padded array to match the dimensions
    of the original array. Inverse operation of zero_padding_2N.
    
    Parameters
    ----------
    padded : np.ndarray
        Padded array
    original : np.ndarray
        Original array (for size reference)
        
    Returns
    -------
    np.ndarray
        Cropped array with same shape as original
        
    Raises
    ------
    ValueError
        If padded array is smaller than original
    """
    _validate_2d_array(padded, "padded")
    _validate_2d_array(original, "original")
    
    H, W = original.shape
    Ph, Pw = padded.shape
    
    if Ph < H or Pw < W:
        raise ValueError(
            f"Padded array ({Ph}×{Pw}) is smaller than original ({H}×{W})"
        )
    
    r0 = (Ph - H) // 2
    c0 = (Pw - W) // 2
    
    return padded[r0:r0 + H, c0:c0 + W]


def crop_to_square(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    margin: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop field and coordinates to a square region around center.
    
    Efficiently extracts a square region of interest from the field, centered
    on the origin, with size determined by radius and margin.
    
    Parameters
    ----------
    field : np.ndarray
        2D complex field
    x, y : np.ndarray
        1D coordinate arrays [m]
    radius : float
        Characteristic radius to include [m]
    margin : float, optional
        Safety margin factor (crop to ±margin*radius), must be >= 1
        
    Returns
    -------
    field_cropped : np.ndarray
        Cropped field
    x_cropped, y_cropped : np.ndarray
        Cropped coordinate arrays
        
    Raises
    ------
    ValueError
        If inputs are invalid or margin < 1
    """
    _validate_2d_array(field, "field")
    _validate_1d_array(x, "x")
    _validate_1d_array(y, "y")
    _validate_coordinate_match(field, x, y)
    _validate_positive(radius, "radius")
    
    if margin < 1.0:
        raise ValueError(f"margin must be >= 1, got {margin}")
    
    lim = margin * radius
    
    # Find indices within crop region - use searchsorted for efficiency
    ix_min = np.searchsorted(x, -lim, side='left')
    ix_max = np.searchsorted(x, lim, side='right')
    iy_min = np.searchsorted(y, -lim, side='left')
    iy_max = np.searchsorted(y, lim, side='right')
    
    # Ensure we have at least some points
    if ix_max <= ix_min or iy_max <= iy_min:
        logger.warning("Crop region is empty, returning original")
        return field, x, y
    
    # Make square by taking minimum dimension
    nx = ix_max - ix_min
    ny = iy_max - iy_min
    n = min(nx, ny)
    
    # Center the square region
    cx = (ix_min + ix_max) // 2
    cy = (iy_min + iy_max) // 2
    h = n // 2
    
    # Compute final slices with bounds checking
    x_start = max(0, cx - h)
    x_end = min(len(x), cx - h + n)
    y_start = max(0, cy - h)
    y_end = min(len(y), cy - h + n)
    
    field_crop = field[y_start:y_end, x_start:x_end]
    x_crop = x[x_start:x_end]
    y_crop = y[y_start:y_end]
    
    return field_crop, x_crop, y_crop


def pad_to_fast_size(
    field: np.ndarray,
    L: float
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Pad field to FFT-friendly size (product of small primes: 2, 3, 5, 7).
    
    Uses scipy's next_fast_len to find optimal FFT size, which significantly
    improves FFT performance by avoiding prime factorizations.
    
    Parameters
    ----------
    field : np.ndarray
        2D input field (must be square)
    L : float
        Physical size of the input field [m]
        
    Returns
    -------
    field_padded : np.ndarray
        Padded field (M × M) where M = next_fast_len(N)
    L_padded : float
        Physical size of padded field [m]
    x, y : np.ndarray
        Coordinate arrays for padded field [m]
        
    Raises
    ------
    ValueError
        If field is not square or L is not positive
    """
    _validate_2d_array(field, "field")
    _validate_positive(L, "L")
    
    if field.shape[0] != field.shape[1]:
        raise ValueError(f"Field must be square, got shape {field.shape}")
    
    N = field.shape[0]
    M = next_fast_len(N)
    
    dx = L / N
    
    if M == N:
        # No padding needed
        coords = (np.arange(N) - N/2 + 0.5) * dx
        return field, L, coords, coords.copy()
    
    pad_total = M - N
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    
    field_padded = np.pad(
        field,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode='constant',
        constant_values=0
    )
    
    # Update physical size and coordinates
    # Keep same pixel spacing, expand physical size
    L_padded = L * M / N
    coords = (np.arange(M) - M/2 + 0.5) * dx
    
    logger.debug(f"Padded from {N}×{N} to {M}×{M} for fast FFT")
    
    return field_padded, L_padded, coords, coords.copy()


# ============================================================================
# SECOND-ORDER MOMENTS
# ============================================================================

def calculate_second_order_moments(
    intensity: np.ndarray,
    dx: float,
    center: Optional[Tuple[float, float]] = None,
    use_center_of_mass: bool = False
) -> Tuple[float, float]:
    """
    Calculate second-order moments of an intensity distribution.
    
    Parameters
    ----------
    intensity : np.ndarray
        2D intensity array
    dx : float
        Pixel spacing [m], must be positive
    center : tuple of float, optional
        (row, col) center position in pixels
        If None, uses array center
    use_center_of_mass : bool, optional
        If True, calculate center of mass and use it as center
        
    Returns
    -------
    d_x, d_y : float
        4σ diameters in x and y directions [m]
        
    Raises
    ------
    ValueError
        If intensity is not 2D or dx is not positive
    """
    _validate_2d_array(intensity, "intensity")
    _validate_positive(dx, "dx")
    
    if use_center_of_mass:
        center_y, center_x = ndimage.center_of_mass(intensity)
    elif center is None:
        center_y = (intensity.shape[0] - 1) / 2.0
        center_x = (intensity.shape[1] - 1) / 2.0
    else:
        center_y, center_x = center
    
    col_indices = np.arange(intensity.shape[1]) - center_x
    row_indices = np.arange(intensity.shape[0]) - center_y
    
    total_intensity = float(np.sum(intensity))
    
    if total_intensity <= _INTENSITY_THRESHOLD:
        logger.warning("Intensity is near zero, cannot calculate moments")
        return 0.0, 0.0
    
    I_sum_rows = np.sum(intensity, axis=0) 
    I_sum_cols = np.sum(intensity, axis=1)  
    
    mom_x = float(np.sum(col_indices**2 * I_sum_rows))
    mom_y = float(np.sum(row_indices**2 * I_sum_cols))
    
    d_x = 4 * np.sqrt(mom_x / total_intensity) * dx
    d_y = 4 * np.sqrt(mom_y / total_intensity) * dx
    
    return d_x, d_y


# ============================================================================
# MASK CREATION
# ============================================================================

def create_circular_mask(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    center: Tuple[float, float] = (0.0, 0.0),
    soft_edge: Optional[float] = None
) -> np.ndarray:
    """
    Create a circular mask with optional soft edges.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    radius : float
        Mask radius [m], must be positive
    center : tuple of float, optional
        (x0, y0) mask center [m]
    soft_edge : float, optional
        If provided, use a smooth sigmoid transition over this width [m]
        Must be positive
        
    Returns
    -------
    np.ndarray
        2D mask (1.0 inside, 0.0 outside, smooth transition if soft_edge)
        
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    _validate_1d_array(x, "x")
    _validate_1d_array(y, "y")
    _validate_positive(radius, "radius")
    
    if soft_edge is not None and soft_edge <= 0:
        raise ValueError(f"soft_edge must be positive, got {soft_edge}")
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='xy')
    X = X - center[0]
    Y = Y - center[1]
    R = np.hypot(X, Y)
    
    if soft_edge is None or soft_edge <= 0:
        return (R <= radius).astype(float)
    
    # Smooth transition using sigmoid
    # Factor of 4 gives ~95% transition over soft_edge width
    transition = 1.0 / (1.0 + np.exp((R - radius) / soft_edge * _SIGMOID_STEEPNESS))
    return transition


def create_rectangular_mask(
    x: np.ndarray,
    y: np.ndarray,
    width: float,
    height: float,
    center: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Create a rectangular mask.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    width, height : float
        Mask dimensions [m], must be positive
    center : tuple of float, optional
        (x0, y0) mask center [m]
        
    Returns
    -------
    np.ndarray
        2D mask (1.0 inside, 0.0 outside)
        
    Raises
    ------
    ValueError
        If width or height are not positive
    """
    _validate_1d_array(x, "x")
    _validate_1d_array(y, "y")
    _validate_positive(width, "width")
    _validate_positive(height, "height")
    
    X, Y = np.meshgrid(x, y, indexing='xy')
    X = X - center[0]
    Y = Y - center[1]
    
    mask_x = np.abs(X) <= width / 2
    mask_y = np.abs(Y) <= height / 2
    
    return (mask_x & mask_y).astype(float)

