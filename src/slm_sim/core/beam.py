import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.special import eval_genlaguerre, eval_hermite
import logging

logger = logging.getLogger(__name__)

# Types alias
IntensityProfile = np.ndarray
PhaseProfile = np.ndarray
ComplexField = np.ndarray

# ============================================================================
# INTENSITY PROFILES
# ============================================================================

def create_intensity_gaussian(
    x: np.ndarray, 
    y: np.ndarray, 
    w0: float, 
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a 2D Gaussian intensity profile.
    
    I(x,y) = exp(-2(x²+y²)/w0²)
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    w0 : float
        1/e² radius [m]
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D intensity profile
    """
    if w0 <= 0:
        raise ValueError(f"Gaussian width must be positive, got {w0}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    R2 = X**2 + Y**2
    
    I = np.exp(-2.0 * R2 / w0**2)
    I[I < 1e-10] = 0.0
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_supergaussian(
    x: np.ndarray,
    y: np.ndarray,
    w0: float,
    order: float = 2.0,
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a super-Gaussian intensity profile.
    
    I(r) = exp(-2(r/w0)^order)
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    w0 : float
        Characteristic radius [m]
    order : float, optional
        Super-Gaussian order (2 = Gaussian, higher = flatter top)
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D super-Gaussian intensity profile
    """
    if w0 <= 0:
        raise ValueError(f"Width must be positive, got {w0}")
    if order <= 0:
        raise ValueError(f"Order must be positive, got {order}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    R = np.hypot(X, Y)
    
    I = np.exp(-2.0 * (R / w0)**order)
    I[I < 1e-10] = 0.0
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_flattop(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a flat-top intensity profile.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    radius : float
        Aperture radius [m]
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D flat-top intensity profile
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    R = np.hypot(X, Y)
    
    I = (R <= radius).astype(float)
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_hollow_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    w0: float,
    n: int = 1,
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a hollow Gaussian beam (HGB) intensity profile.
    
    E(r) ∝ (r/w0)^n * exp(-r²/w0²)
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    w0 : float
        Characteristic radius [m]
    n : int, optional
        Hollow order (higher = wider dark center)
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D hollow Gaussian intensity profile
    """
    if w0 <= 0:
        raise ValueError(f"Width must be positive, got {w0}")
    if n < 0:
        raise ValueError(f"Order must be non-negative, got {n}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    R2 = X**2 + Y**2
    
    E = (R2 / w0**2)**n * np.exp(-R2 / w0**2)
    I = np.abs(E)**2
    I[I < 1e-10] = 0.0
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_laguerre_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    w0: float,
    p: int = 0,
    l: int = 0,
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a Laguerre-Gaussian (LG_p^l) beam intensity at z=0.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    w0 : float
        Beam waist [m]
    p : int, optional
        Radial mode index (≥0)
    l : int, optional
        Azimuthal mode index (orbital angular momentum)
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D Laguerre-Gaussian intensity profile
        
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    r2 = X**2 + Y**2
    theta = np.arctan2(Y, X)
    
    u = 2.0 * r2 / w0**2
    radial_part = (np.sqrt(2 * r2) / w0)**abs(l)
    laguerre = eval_genlaguerre(p, abs(l), u)
    envelope = np.exp(-r2 / w0**2)
    
    E = radial_part * laguerre * envelope * np.exp(1j * l * theta)
    I = np.abs(E)**2
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_hermite_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    w0: float,
    m: int = 0,
    n: int = 0,
    center: Tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a Hermite-Gaussian (HG_mn) beam intensity at z=0.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    w0 : float
        Beam waist [m]
    m, n : int, optional
        Mode indices in x and y directions (≥0)
    center : tuple of float, optional
        (x0, y0) center position [m]
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D Hermite-Gaussian intensity profile
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    
    xi = np.sqrt(2) * X / w0
    eta = np.sqrt(2) * Y / w0
    
    Hm = eval_hermite(m, xi)
    Hn = eval_hermite(n, eta)
    envelope = np.exp(-(X**2 + Y**2) / w0**2)
    
    E = Hm * Hn * envelope
    I = np.abs(E)**2
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


def create_intensity_flat(
    x: np.ndarray,
    y: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True
) -> IntensityProfile:
    """
    Create a uniform (flat) intensity profile.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    mask : np.ndarray, optional
        Boolean mask to apply
    normalize : bool, optional
        If True, normalize to unit total intensity
        
    Returns
    -------
    np.ndarray
        2D flat intensity profile
    """
    I = np.ones((len(y), len(x)), dtype=float)
    
    if mask is not None:
        I *= mask.astype(float)
    
    if normalize:
        total = I.sum()
        if total > 1e-30:
            I = I / total
    
    return I


# ============================================================================
# PHASE PROFILES
# ============================================================================

def create_phase_vortex(
    x: np.ndarray,
    y: np.ndarray,
    charge: int,
    center: Tuple[float, float] = (0.0, 0.0)
) -> PhaseProfile:
    """
    Create an optical vortex (spiral) phase.
    
    φ(θ) = l * θ, where l is the topological charge.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    charge : int
        Topological charge (orbital angular momentum quantum number)
    center : tuple of float, optional
        (x0, y0) vortex center [m]
        
    Returns
    -------
    np.ndarray
        2D vortex phase profile [radians], in [0, 2π)
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    theta = np.arctan2(Y, X)
    
    phi = charge * theta
    return np.mod(phi, 2 * np.pi)


def create_phase_binary(
    x: np.ndarray,
    y: np.ndarray,
    stripe_width: float,
    phase_shift: float = np.pi,
    angle_deg: float = 0.0,
    center: Tuple[float, float] = (0.0, 0.0)
) -> PhaseProfile:
    """
    Create a binary striped phase pattern.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    stripe_width : float
        Width of each stripe [m]
    phase_shift : float, optional
        Phase difference between stripes [radians]
    angle_deg : float, optional
        Stripe rotation angle [degrees]
    center : tuple of float, optional
        (x0, y0) pattern center [m]
        
    Returns
    -------
    np.ndarray
        2D binary phase pattern [radians], in [0, 2π)
    """
    if stripe_width <= 0:
        raise ValueError(f"Stripe width must be positive, got {stripe_width}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    
    angle_rad = np.deg2rad(angle_deg)
    coordinate = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    
    stripe_index = np.floor(coordinate / stripe_width).astype(int)
    phi = np.where(stripe_index % 2 == 0, 0.0, phase_shift)
    
    return np.mod(phi, 2 * np.pi)


def create_phase_radial_step(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    inner_phase: float = 0.0,
    outer_phase: float = np.pi,
    center: Tuple[float, float] = (0.0, 0.0)
) -> PhaseProfile:
    """
    Create a radial phase step at a given radius.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    radius : float
        Step radius [m]
    inner_phase : float, optional
        Phase inside the radius [radians]
    outer_phase : float, optional
        Phase outside the radius [radians]
    center : tuple of float, optional
        (x0, y0) center position [m]
        
    Returns
    -------
    np.ndarray
        2D radial phase step [radians], in [0, 2π)
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X - center[0]
    Y = Y - center[1]
    R = np.hypot(X, Y)
    
    phi = np.where(R <= radius, inner_phase, outer_phase)
    
    return np.mod(phi, 2 * np.pi)


def create_phase_two_foci_checkerboard(
    x: np.ndarray,
    y: np.ndarray,
    wavelength: float,
    focal_length: float,
    separation: float,
    phase_difference: float = 0.0,
    checker_pitch: float = 60e-6,
    angle_deg: float = 0.0,
    tilt_a: bool = True,
    tilt_b: bool = True
) -> PhaseProfile:
    """
    Create a checkerboard pattern for generating two separated foci.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    wavelength : float
        Wavelength [m]
    focal_length : float
        Focal length [m]
    separation : float
        Separation between foci at focal plane [m]
    phase_difference : float, optional
        Phase difference between two foci [radians]
    checker_pitch : float, optional
        Size of checkerboard squares [m]
    angle_deg : float, optional
        Rotation angle [degrees]
    tilt_a : bool, optional
        Apply tilt to beam A
    tilt_b : bool, optional
        Apply tilt to beam B
        
    Returns
    -------
    np.ndarray
        2D two-foci phase pattern [radians], in [0, 2π)
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    
    k0 = 2 * np.pi / wavelength
    theta = separation / (2 * focal_length)
    k_t = k0 * theta
    
    angle_rad = np.deg2rad(angle_deg)
    U = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    
    # Phase for beam A
    if tilt_a:
        phi_A = +k_t * U
    else:
        phi_A = 0.0
    
    # Phase for beam B
    if tilt_b:
        phi_B = -k_t * U + phase_difference
    else:
        phi_B = phase_difference
    
    # Checkerboard pattern
    x_min, y_min = float(x[0]), float(y[0])
    ix = np.floor((X - x_min) / checker_pitch).astype(int)
    iy = np.floor((Y - y_min) / checker_pitch).astype(int)
    checker = (ix + iy) & 1
    
    phi = np.where(checker == 0, phi_A, phi_B)
    return np.mod(phi, 2 * np.pi)



def create_phase_two_foci_stochastic(
    x: np.ndarray,
    y: np.ndarray,
    wavelength: float,
    focal_length: float,
    separation: float,
    A_rel_B: float = 0.5,
    phase_difference: float = 0.0,
    angle_deg: float = 0.0,
    tilt_a: bool = True,
    tilt_b: bool = True,
    seed: int = 12345
) -> PhaseProfile:
    """
    Create a stochastic pattern for generating two separated foci.
    
    Each pixel is randomly assigned to beam A or B based on the A_rel_B ratio.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays [m]
    wavelength : float
        Wavelength [m]
    focal_length : float
        Focal length [m]
    separation : float
        Separation between foci at focal plane [m]
    A_rel_B : float, optional
        Relative power ratio A/(A+B). 
        - 0.0: all power in B
        - 0.5: equal power split (50/50)
        - 1.0: all power in A
    phase_difference : float, optional
        Phase difference between two foci [radians]
    angle_deg : float, optional
        Rotation angle [degrees]
    tilt_a : bool, optional
        Apply tilt to beam A
    tilt_b : bool, optional
        Apply tilt to beam B
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        2D two-foci stochastic phase pattern [radians], in [0, 2π)
        
    Notes
    -----
    The stochastic pattern randomly assigns each pixel to either beam A or B
    with probability proportional to A_rel_B. This creates a spatially random
    but statistically well-defined power splitting between the two foci.
    """
    if not 0.0 <= A_rel_B <= 1.0:
        raise ValueError(f"A_rel_B must be between 0 and 1, got {A_rel_B}")
    
    X, Y = np.meshgrid(x, y, indexing="xy")
    
    k0 = 2 * np.pi / wavelength
    theta = separation / (2 * focal_length)
    k_t = k0 * theta
    
    angle_rad = np.deg2rad(angle_deg)
    U = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    
    # Phase for beam A
    if tilt_a:
        phi_A = +k_t * U
    else:
        phi_A = 0.0
    
    # Phase for beam B
    if tilt_b:
        phi_B = -k_t * U + phase_difference
    else:
        phi_B = phase_difference
    
    # Stochastic pattern
    np.random.seed(seed)
    
    random_mask = np.random.rand(*X.shape)
    stochastic_pattern = random_mask < A_rel_B  # True for A, False for B
    
    phi = np.where(stochastic_pattern, phi_A, phi_B)
    return np.mod(phi, 2 * np.pi)


def quantize_phase(
    phase: np.ndarray,
    bits: int = 8,
    phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
    wrap: bool = True
) -> PhaseProfile:
    """
    Quantize a continuous phase to discrete levels.
    
    Simulates the effect of a digital SLM with limited phase resolution.
    
    Parameters
    ----------
    phase : np.ndarray
        Continuous phase array [radians]
    bits : int, optional
        Number of quantization bits (2^bits levels)
    phase_range : tuple of float, optional
        (min, max) phase range [radians]
    wrap : bool, optional
        If True, wrap input phase to phase_range before quantizing
        
    Returns
    -------
    np.ndarray
        Quantized phase array [radians]
        
    Examples
    --------
    >>> phi_8bit = quantize_phase(phi_continuous, bits=8)  # 256 levels
    >>> phi_10bit = quantize_phase(phi_continuous, bits=10)  # 1024 levels
    """
    if bits < 1:
        raise ValueError(f"Number of bits must be ≥1, got {bits}")
    
    vmin, vmax = phase_range
    width = vmax - vmin
    
    if wrap:
        phase = (phase - vmin) % width + vmin
    else:
        phase = np.clip(phase, vmin, vmax - np.finfo(float).eps)
    
    levels = 2**bits
    step = width / (levels - 1)
    
    quantized = np.round((phase - vmin) / step) * step + vmin
    quantized = np.clip(quantized, vmin, vmax - np.finfo(float).eps)
    
    return quantized