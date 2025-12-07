import numpy as np
from scipy import fft as sfft
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

_VALID_NORMS = {"ortho", "forward", "backward"}


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def _validate_field(field: np.ndarray, name: str = "field") -> None:
    """Validate field array."""
    if not isinstance(field, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(field)}")
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {field.ndim}D with shape {field.shape}")


def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_norm(norm: str) -> None:
    """Validate FFT normalization parameter."""
    if norm not in _VALID_NORMS:
        raise ValueError(
            f"norm must be one of {_VALID_NORMS}, got '{norm}'"
        )


# ============================================================================
# CENTERED FFT OPERATIONS
# ============================================================================

def fft2c(
    field: np.ndarray,
    workers: int = -1,
    norm: str = "ortho"
) -> np.ndarray:
    """
    Centered 2D FFT with orthonormal normalization.
    
    Performs: fftshift(fft2(ifftshift(field)))
    
    This is the standard centered FFT used in optics, where the zero-frequency
    component is shifted to the center of the spectrum.
    
    Parameters
    ----------
    field : np.ndarray
        2D input field (real or complex)
    workers : int, optional
        Number of workers for parallel FFT (-1 uses all cores)
    norm : str, optional
        Normalization mode:
        - 'ortho': Orthonormal (default, energy preserving)
        - 'forward': 1/n normalization on forward transform
        - 'backward': 1/n normalization on inverse transform
        
    Returns
    -------
    np.ndarray
        2D Fourier transform (centered, same dtype as input)
    """
    _validate_field(field, "field")
    _validate_norm(norm)
    
    return np.fft.fftshift(
        sfft.fft2(np.fft.ifftshift(field), norm=norm, workers=workers)
    )


def ifft2c(
    field: np.ndarray,
    workers: int = -1,
    norm: str = "ortho"
) -> np.ndarray:
    """
    Centered 2D inverse FFT with orthonormal normalization.
    
    Performs: fftshift(ifft2(ifftshift(field)))
    
    Inverse operation of fft2c, used to transform from frequency domain
    back to spatial domain.
    
    Parameters
    ----------
    field : np.ndarray
        2D input field (typically in frequency domain)
    workers : int, optional
        Number of workers for parallel FFT (-1 uses all cores)
    norm : str, optional
        Normalization mode ('ortho', 'forward', 'backward')
        
    Returns
    -------
    np.ndarray
        2D inverse Fourier transform (centered)
        
    Raises
    ------
    ValueError
        If field is not 2D or norm is invalid
    """
    _validate_field(field, "field")
    _validate_norm(norm)
    
    return np.fft.fftshift(
        sfft.ifft2(np.fft.ifftshift(field), norm=norm, workers=workers)
    )


# ============================================================================
# FREQUENCY GRID GENERATION
# ============================================================================

def create_frequency_grid(
    N: int,
    L: float,
    centered: bool = True
) -> np.ndarray:
    """
    Create a 1D frequency grid for FFT.
    
    Parameters
    ----------
    N : int
        Number of points, must be >= 2
    L : float
        Physical size [m], must be positive
    centered : bool, optional
        If True, return centered frequencies (for use with fft2c/ifft2c)
        If False, return standard FFT frequency ordering
        
    Returns
    -------
    np.ndarray
        Frequency array [1/m]
        
    Raises
    ------
    ValueError
        If N < 2 or L <= 0
        
    Notes
    -----
    Frequency spacing is Δf = 1/L.
    Maximum frequency (Nyquist) is ±N/(2L) for centered grids.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    _validate_positive(L, "L")
    
    df = 1.0 / L
    
    if centered:
        freqs = (np.arange(N) - N/2) * df
    else:
        freqs = np.fft.fftfreq(N, d=L/N)
    
    return freqs


def create_frequency_grid_2d(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    centered: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 2D frequency grids for FFT.
    
    Parameters
    ----------
    Nx, Ny : int
        Number of points in x and y, must be >= 2
    Lx, Ly : float
        Physical sizes [m], must be positive
    centered : bool, optional
        If True, return centered frequencies (for use with fft2c)
        
    Returns
    -------
    Fx, Fy : np.ndarray
        2D frequency grids [1/m], shape (Ny, Nx)
        
    Raises
    ------
    ValueError
        If any dimension < 2 or size <= 0
        
    Notes
    -----
    The grids are created with indexing='xy', meaning:
    - Fx varies along columns (x-direction)
    - Fy varies along rows (y-direction)
    """
    fx = create_frequency_grid(Nx, Lx, centered=centered)
    fy = create_frequency_grid(Ny, Ly, centered=centered)
    
    return np.meshgrid(fx, fy, indexing='xy')


# ============================================================================
# FILTERING
# ============================================================================

def bandlimit_filter(
    Fx: np.ndarray,
    Fy: np.ndarray,
    max_freq: float,
    soft_edge: bool = False,
    transition_width: float = 0.1
) -> np.ndarray:
    """
    Create a circular bandlimit filter for frequency domain.
    
    Used to remove high frequencies that cannot propagate (evanescent waves)
    
    Parameters
    ----------
    Fx, Fy : np.ndarray
        2D frequency grids [1/m]
    max_freq : float
        Maximum frequency [1/m], must be positive
    soft_edge : bool, optional
        If True, use soft (Gaussian) edge instead of hard cutoff
    transition_width : float, optional
        Width of transition region as fraction of max_freq (0-1)
        Only used if soft_edge=True
        
    Returns
    -------
    np.ndarray
        2D bandlimit filter (1.0 inside circle, 0.0 outside)
        
    Raises
    ------
    ValueError
        If max_freq <= 0 or transition_width not in (0, 1)
    """
    _validate_positive(max_freq, "max_freq")
    
    if soft_edge and not 0 < transition_width < 1:
        raise ValueError(f"transition_width must be in (0, 1), got {transition_width}")
    
    F = np.sqrt(Fx**2 + Fy**2)
    
    if not soft_edge:
        # Hard circular cutoff
        return (F <= max_freq).astype(float)
    else:
        # Soft edge using super-Gaussian
        mask = np.ones_like(F)
        beyond_cutoff = F > max_freq
        
        if np.any(beyond_cutoff):
            width = transition_width * max_freq
            # Gaussian rolloff
            mask[beyond_cutoff] = np.exp(
                -((F[beyond_cutoff] - max_freq) / width)**2
            )
        
        return mask