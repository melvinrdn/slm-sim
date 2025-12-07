import time
import numpy as np
from typing import Tuple, Optional, Literal, Union
import logging

from ..utils.fft import fft2c, ifft2c, create_frequency_grid_2d, bandlimit_filter
from ..utils.arrays import (
    calculate_second_order_moments,
    crop_to_square,
    pad_to_fast_size,
)
from .physics import calculate_energy

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Numerical thresholds
_ENERGY_THRESHOLD = 1e-30
_DEFAULT_ENERGY_TOLERANCE = 0.05
_MIN_DISTANCE_FOR_BANDLIMIT = 1e-12

# Default data types for performance
_COMPLEX_DTYPE = np.complex64
_FLOAT_DTYPE = np.float32


# ============================================================================
# INPUT VALIDATION HELPERS
# ============================================================================

def _validate_field(field: np.ndarray, name: str = "field") -> None:
    """Validate that field is a 2D array."""
    if not isinstance(field, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {field.shape}")
    if field.shape[0] != field.shape[1]:
        raise ValueError(f"{name} must be square, got shape {field.shape}")


def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

# ============================================================================
# COORDINATE SYSTEM DEFINITIONS
# ============================================================================

def create_slm_coordinates(
    size: float,
    num_points: int,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create coordinate system for SLM plane.
    
    Parameters
    ----------
    size : float
        Physical size of SLM plane (assumed square) [m]
    num_points : int
        Number of sampling points (N x N grid)
    verbose : bool, optional
        If True, log coordinate information
        
    Returns
    -------
    x, y : np.ndarray
        1D coordinate arrays [m]
    extent : list
        [x_min, x_max, y_min, y_max] for plotting [m]
    """
    _validate_positive(size, "size")
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    
    N = int(num_points)
    dx = 2.0 * size / N
    
    indices = np.arange(N)
    x = (indices - N/2 + 0.5) * dx
    y = x.copy()
    
    extent = [x[0], x[-1], y[0], y[-1]]
    
    if verbose:
        logger.info(
            f"SLM plane: [{extent[0]*1e3:.2f}, {extent[1]*1e3:.2f}] mm × "
            f"[{extent[2]*1e3:.2f}, {extent[3]*1e3:.2f}] mm, "
            f"{N}×{N} points, dx={dx*1e6:.2f} µm"
        )
    
    return x, y, extent


def create_focal_plane_coordinates(
    size_slm: float,
    num_points: int,
    focal_length: float,
    wavelength: float,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create coordinate system for focal plane (after Fourier transform).
    
    Parameters
    ----------
    size_slm : float
        Physical size of SLM plane [m]
    num_points : int
        Number of sampling points
    focal_length : float
        Focal length [m]
    wavelength : float
        Wavelength [m]
    verbose : bool, optional
        If True, log coordinate information
        
    Returns
    -------
    x, y : np.ndarray
        1D coordinate arrays [m]
    extent : list
        [x_min, x_max, y_min, y_max] for plotting [m]
    """
    _validate_positive(size_slm, "size_slm")
    _validate_positive(focal_length, "focal_length")
    _validate_positive(wavelength, "wavelength")
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    
    N = int(num_points)
    k0 = 2 * np.pi / wavelength
    
    # Frequency spacing in k-space
    dkx = np.pi / (2.0 * size_slm)
    
    # Convert to focal plane coordinates
    indices = np.arange(N)
    k_x = (indices - N/2 + 0.5) * dkx
    x = focal_length * k_x / k0
    y = x.copy()
    
    extent = [x[0], x[-1], y[0], y[-1]]
    
    if verbose:
        logger.info(
            f"Focal plane: [{extent[0]*1e6:.2f}, {extent[1]*1e6:.2f}] µm × "
            f"[{extent[2]*1e6:.2f}, {extent[3]*1e6:.2f}] µm, "
            f"{N}×{N} points"
        )
    
    return x, y, extent


# ============================================================================
# INTERNAL PROPAGATION METHODS (NOT PUBLIC API)
# ============================================================================

def _propagate_fourier(
    field: np.ndarray,
    direction: Literal["forward", "backward"] = "forward",
    workers: int = -1
) -> np.ndarray:
    """
    Propagate field using Fourier transform (thin lens / far-field approximation).
    
    This implements the relationship between an SLM plane and its focal plane
    through a lens, where the focal plane field is the Fourier transform of
    the SLM plane field.
    
    Parameters
    ----------
    field : np.ndarray
        2D complex field
    direction : str, optional
        'forward' for SLM → focus, 'backward' for focus → SLM
    workers : int, optional
        Number of parallel workers for FFT
        
    Returns
    -------
    np.ndarray
        Propagated field
        
    Notes
    -----
    This is an internal function. Use `propagate()` with method='fourier' instead.
    """
    _validate_field(field, "field")
    
    if direction == "forward":
        return fft2c(field, workers=workers)
    elif direction == "backward":
        return ifft2c(field, workers=workers)
    else:
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")


def _propagate_angular_spectrum(
    field: np.ndarray,
    distance: float,
    wavelength: float,
    L: float,
    bandlimit: bool = True,
    workers: int = -1
) -> np.ndarray:
    """
    Propagate field using angular spectrum method.
    
    Implements the Rayleigh-Sommerfeld diffraction integral via
    angular spectrum propagation (ASPW).
    
    Parameters
    ----------
    field : np.ndarray
        2D input complex field
    distance : float
        Propagation distance [m] (can be negative)
    wavelength : float
        Wavelength [m]
    L : float
        Physical size of the field [m]
    aperture_size : float, optional
        Physical aperture size [m] for Nyquist check
        If None, uses L
    bandlimit : bool, optional
        If True, apply bandlimiting filter to avoid aliasing
    workers : int, optional
        Number of parallel workers for FFT
        
    Returns
    -------
    np.ndarray
        Propagated field
        
    Notes
    -----
    This is an internal function. Use `propagate()` with method='angular_spectrum' instead.
    """
    _validate_field(field, "field")
    _validate_positive(wavelength, "wavelength")
    _validate_positive(L, "L")
    
    N = field.shape[0]
    k = 2 * np.pi / wavelength
    dx = L / N
    
    # Create frequency grid
    Fx, Fy = create_frequency_grid_2d(N, N, L, L, centered=True)
    
    # Transfer function (propagator)
    # sqrt(1 - (λfx)² - (λfy)²) is the z-component of normalized k-vector
    kz_norm = np.sqrt(
        1.0 - (wavelength * Fx)**2 - (wavelength * Fy)**2,
        dtype=complex  # Allow complex for evanescent waves
    )
    
    transfer_function = np.exp(1j * k * distance * kz_norm)
    
    # Apply bandlimit if requested
    if bandlimit:
        # Maximum spatial frequency to avoid evanescent waves
        # This occurs when kz_norm becomes imaginary
        f_max = 1 / (wavelength * np.sqrt(1 + (2 * distance / L)**2))
        bandlimit_mask = bandlimit_filter(Fx, Fy, f_max)
        transfer_function *= bandlimit_mask
    
    # Propagate: FFT → multiply by transfer function → inverse FFT
    field_fft = fft2c(field, workers=workers)
    field_fft_propagated = field_fft * transfer_function
    field_propagated = ifft2c(field_fft_propagated, workers=workers)
    
    return field_propagated


# ============================================================================
# Z-SCAN HELPER FUNCTIONS
# ============================================================================

def _prepare_field_for_zscan(
    field_focus: np.ndarray,
    x_focus: np.ndarray,
    y_focus: np.ndarray,
    crop_margin: float,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Prepare field for z-scan by cropping and padding.
    
    Returns
    -------
    field_padded : np.ndarray
        Padded field ready for propagation
    x_final, y_final : np.ndarray
        Final coordinate arrays
    L_padded : float
        Physical size of padded field
    """
    # Calculate physical size and pixel spacing
    L = x_focus[-1] - x_focus[0]
    dx = L / (len(x_focus) - 1) if len(x_focus) > 1 else L
    
    # Calculate beam size for cropping
    intensity = np.abs(field_focus)**2
    d_x, d_y = calculate_second_order_moments(intensity, dx, use_center_of_mass=True)
    beam_radius = max(d_x, d_y)
    
    if verbose:
        logger.info(f"Beam diameter: {beam_radius*1e6:.2f} µm (4σ)")
    
    # Crop to relevant region
    field_crop, x_crop, y_crop = crop_to_square(
        field_focus, x_focus, y_focus, beam_radius, margin=crop_margin
    )
    
    N_crop = field_crop.shape[0]
    L_crop = x_crop[-1] - x_crop[0]
    
    # Pad to FFT-friendly size
    field_padded, L_padded, x_final, y_final = pad_to_fast_size(field_crop, L_crop)
    N_final = field_padded.shape[0]
    
    if verbose:
        logger.info(f"Cropped to {N_crop}×{N_crop}, padded to {N_final}×{N_final}")
    
    # Convert to optimized dtype
    field_padded = field_padded.astype(_COMPLEX_DTYPE)
    
    return field_padded, x_final, y_final, L_padded


def _create_transfer_function(
    N: int,
    L: float,
    wavelength: float,
    z0: float,
    bandlimit: bool,
    z_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create transfer function for angular spectrum propagation.
    
    Returns
    -------
    kz_norm : np.ndarray
        Normalized z-component of k-vector
    W : np.ndarray
        Bandlimit filter (scalar 1.0 if bandlimit=False)
    """
    k0 = 2 * np.pi / wavelength
    
    # Create frequency grid
    Fx, Fy = create_frequency_grid_2d(N, N, L, L, centered=True)
    
    # Compute transfer function components
    kz_norm = np.sqrt(
        1.0 - (wavelength * Fx)**2 - (wavelength * Fy)**2,
        dtype=_FLOAT_DTYPE
    ).real  # Only keep real part (ignore evanescent)
    
    # Apply bandlimit if requested
    if bandlimit:
        dz_mid = 0.5 * (abs(z_range[0]) + abs(z_range[1]))
        dz_mid = max(dz_mid, _MIN_DISTANCE_FOR_BANDLIMIT)
        f_max = L / (wavelength * np.sqrt(L**2 + 4 * dz_mid**2))
        W = bandlimit_filter(Fx, Fy, f_max).astype(_FLOAT_DTYPE)
    else:
        W = _FLOAT_DTYPE(1.0)
    
    return kz_norm, W


def _propagate_zscan_loop(
    field_fft: np.ndarray,
    z: np.ndarray,
    kz_norm: np.ndarray,
    W: Union[np.ndarray, float],
    k0: float,
    workers: int,
    num_steps: int,
    N: int
) -> np.ndarray:
    """
    Perform the actual z-scan propagation loop.
    
    Returns
    -------
    fields_z : np.ndarray
        3D array of propagated fields
    """
    # Initialize output array
    fields_z = np.empty((N, N, num_steps), dtype=_COMPLEX_DTYPE)
    
    if num_steps >= 2:
        dz = z[1] - z[0]
        
        # Initial transfer function at z[0]
        H = np.exp(1j * k0 * _FLOAT_DTYPE(z[0]) * kz_norm).astype(_COMPLEX_DTYPE) * W
        
        # Incremental update factor
        H_increment = np.exp(1j * k0 * _FLOAT_DTYPE(dz) * kz_norm).astype(_COMPLEX_DTYPE)
        
        # Carrier wave (plane wave phase)
        carrier = np.exp(-1j * k0 * z[0]).astype(_COMPLEX_DTYPE)
        carrier_increment = np.exp(-1j * k0 * dz).astype(_COMPLEX_DTYPE)
        
        for i in range(num_steps):
            # Propagate and add carrier wave
            fields_z[:, :, i] = carrier * ifft2c(field_fft * H, workers=workers)
            
            # Update for next step
            H *= H_increment
            carrier *= carrier_increment
    else:
        # Single step
        H = np.exp(1j * k0 * _FLOAT_DTYPE(z[0]) * kz_norm).astype(_COMPLEX_DTYPE) * W
        carrier = np.exp(-1j * k0 * z[0]).astype(_COMPLEX_DTYPE)
        fields_z[:, :, 0] = carrier * ifft2c(field_fft * H, workers=workers)
    
    return fields_z


# ============================================================================
# PUBLIC PROPAGATION API
# ============================================================================

def propagate(
    field: np.ndarray,
    method: Literal["fourier", "angular_spectrum"],
    wavelength: float,
    L: float,
    distance: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Unified interface for field propagation.
    
    Parameters
    ----------
    field : np.ndarray
        2D input complex field (must be square)
    method : str
        Propagation method: 'fourier' or 'angular_spectrum'
    wavelength : float
        Wavelength [m], must be positive
    L : float
        Physical size of the field [m], must be positive
    distance : float, optional
        Propagation distance [m] (required for angular_spectrum)
    **kwargs
        Additional arguments passed to specific propagation method:
        - direction : str (for fourier method)
        - bandlimit : bool (for angular_spectrum method)
        - workers : int (for both methods)
        
    Returns
    -------
    np.ndarray
        Propagated field
    """
    _validate_field(field, "field")
    _validate_positive(wavelength, "wavelength")
    _validate_positive(L, "L")
    
    if method == "fourier":
        return _propagate_fourier(field, **kwargs)
    elif method == "angular_spectrum":
        if distance is None:
            raise ValueError("distance must be provided for angular_spectrum method")
        return _propagate_angular_spectrum(
            field, distance, wavelength, L
            , **kwargs
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Must be 'fourier' or 'angular_spectrum'"
        )


def propagate_z_scan(
    field_focus: np.ndarray,
    z_range: Tuple[float, float],
    num_steps: int,
    wavelength: float,
    rayleigh_range: float,
    size_slm: float,
    num_points_slm: int,
    bandlimit: bool = True,
    workers: int = -1,
    crop_margin: float = 2.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a z-scan by propagating field through multiple planes.
    
    Parameters
    ----------
    field_focus : np.ndarray
        Complex field at focal plane (z=0), must be square 2D array
    z_range : tuple of float
        (z_min, z_max) in units of Rayleigh range [zR]
    num_steps : int
        Number of z-planes to compute, must be >= 1
    wavelength : float
        Wavelength [m], must be positive
    rayleigh_range : float
        Rayleigh range [m], must be positive
    size_slm : float
        Physical size of SLM plane [m], must be positive
    num_points_slm : int
        Number of points in SLM plane, must be >= 2
    bandlimit : bool, optional
        Apply bandlimiting filter to avoid aliasing
    workers : int, optional
        Number of parallel workers for FFT (-1 uses all cores)
    crop_margin : float, optional
        Crop field to margin*beam_radius before propagation (must be >= 1)
    verbose : bool, optional
        Log progress information
        
    Returns
    -------
    fields_z : np.ndarray
        3D array of shape (Ny, Nx, Nz) containing field at each z-plane
    z : np.ndarray
        1D array of z-positions [m]
    x, y : np.ndarray
        1D coordinate arrays for the cropped field [m]
    """
    # Input validation
    _validate_field(field_focus, "field_focus")
    _validate_positive(wavelength, "wavelength")
    _validate_positive(rayleigh_range, "rayleigh_range")
    _validate_positive(size_slm, "size_slm")
    
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if num_points_slm < 2:
        raise ValueError(f"num_points_slm must be >= 2, got {num_points_slm}")
    if crop_margin < 1.0:
        raise ValueError(f"crop_margin must be >= 1, got {crop_margin}")
    if z_range[0] >= z_range[1]:
        raise ValueError(f"z_range[0] must be < z_range[1], got {z_range}")
    
    t_start = time.perf_counter()
    
    # Create z-array in physical units
    z = np.linspace(
        z_range[0] * rayleigh_range,
        z_range[1] * rayleigh_range,
        num_steps
    )
    
    # Get focal plane coordinates
    x_focus, y_focus, _ = create_focal_plane_coordinates(
        size_slm, num_points_slm, 
        1.0,
        wavelength,
        verbose=False
    )
    
    # Prepare field: crop and pad
    field_padded, x_final, y_final, L_padded = _prepare_field_for_zscan(
        field_focus, x_focus, y_focus, crop_margin, verbose
    )
    N_final = field_padded.shape[0]
    
    # Create transfer function components
    k0 = 2 * np.pi / wavelength
    kz_norm, W = _create_transfer_function(
        N_final, L_padded, wavelength, z[0], bandlimit, 
        (z.min(), z.max())
    )
    
    # Compute FFT of initial field
    field_fft = fft2c(field_padded, workers=workers).astype(_COMPLEX_DTYPE)
    
    # Propagation loop
    t_prop_start = time.perf_counter()
    
    fields_z = _propagate_zscan_loop(
        field_fft, z, kz_norm, W, k0, workers, num_steps, N_final
    )
    
    t_prop = time.perf_counter() - t_prop_start
    t_total = time.perf_counter() - t_start
    
    if verbose:
        logger.info(f"Propagation loop: {t_prop:.3f} s")
        logger.info(f"Total z-scan time: {t_total:.3f} s (including setup)")
    
    return fields_z, z, x_final, y_final


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_energy_conservation(
    field_in: np.ndarray,
    field_out: np.ndarray,
    tolerance: float = _DEFAULT_ENERGY_TOLERANCE,
    method_name: str = "propagation"
) -> bool:
    """
    Check if energy is conserved during propagation.
    
    Parameters
    ----------
    field_in : np.ndarray
        Input field (2D complex array)
    field_out : np.ndarray
        Output field (2D complex array)
    tolerance : float, optional
        Relative tolerance for energy mismatch (default: 0.05 = 5%)
    method_name : str, optional
        Name of method (for logging)
        
    Returns
    -------
    bool
        True if energy is conserved within tolerance
    """
    _validate_field(field_in, "field_in")
    _validate_field(field_out, "field_out")
    
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    
    if field_in.shape != field_out.shape:
        raise ValueError(
            f"Field shapes must match: field_in={field_in.shape}, "
            f"field_out={field_out.shape}"
        )
    
    E_in = calculate_energy(field_in)
    E_out = calculate_energy(field_out)
    
    if E_in < _ENERGY_THRESHOLD:
        logger.warning(f"{method_name}: Input field has near-zero energy")
        return True
    
    relative_error = abs(E_out - E_in) / E_in
    
    if relative_error > tolerance:
        logger.warning(
            f"{method_name}: Energy mismatch {relative_error*100:.2f}% "
            f"(Ein={E_in:.2e}, Eout={E_out:.2e})"
        )
        return False
    
    return True