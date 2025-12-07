import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

_ENERGY_THRESHOLD = 1e-30


# ============================================================================
# BEAM PROPAGATION FORMULAS
# ============================================================================

def calculate_rayleigh_range(w0: float, wavelength: float) -> float:
    """
    Calculate the Rayleigh range for a Gaussian beam.
    
    Parameters
    ----------
    w0 : float
        Beam waist radius [m]
    wavelength : float
        Wavelength [m]
        
    Returns
    -------
    float
        Rayleigh range [m]
    """
    if w0 <= 0:
        raise ValueError(f"Beam waist must be positive, got {w0}")
    if wavelength <= 0:
        raise ValueError(f"Wavelength must be positive, got {wavelength}")
    
    return np.pi * w0**2 / wavelength


def calculate_focused_beam_waist(
    w_lens: float, 
    wavelength: float, 
    focal_length: float
) -> float:
    """
    Calculate the focused beam waist after a lens.
    
    Uses the thin lens approximation for a collimated Gaussian beam.
    
    Parameters
    ----------
    w_lens : float
        Beam radius at the lens [m]
    wavelength : float
        Wavelength [m]
    focal_length : float
        Focal length of the lens [m]
        
    Returns
    -------
    float
        Focused beam waist radius [m]
    """
    if w_lens <= 0:
        raise ValueError(f"Beam radius at lens must be positive, got {w_lens}")
    if wavelength <= 0:
        raise ValueError(f"Wavelength must be positive, got {wavelength}")
    if focal_length <= 0:
        raise ValueError(f"Focal length must be positive, got {focal_length}")
    
    return wavelength * focal_length / (np.pi * w_lens)


def calculate_beam_radius(w0: float, z: float, zR: float) -> float:
    """
    Calculate the beam radius at distance z from waist.
    
    Parameters
    ----------
    w0 : float
        Beam waist radius [m]
    z : float
        Distance from waist [m]
    zR : float
        Rayleigh range [m]
        
    Returns
    -------
    float
        Beam radius at distance z [m]
    """
    return w0 * np.sqrt(1 + (z / zR)**2)


def calculate_divergence_angle(w0: float, wavelength: float) -> float:
    """
    Calculate the far-field divergence half-angle of a Gaussian beam.
    
    Parameters
    ----------
    w0 : float
        Beam waist radius [m]
    wavelength : float
        Wavelength [m]
        
    Returns
    -------
    float
        Divergence half-angle [radians]
    """
    return wavelength / (np.pi * w0)


def calculate_gouy_phase(z: float, zR: float) -> float:
    """
    Calculate the Gouy phase shift at distance z.
    
    Parameters
    ----------
    z : float
        Distance from waist [m]
    zR : float
        Rayleigh range [m]
        
    Returns
    -------
    float
        Gouy phase [radians]
    """
    return np.arctan(z / zR)


def calculate_beam_parameters(
    wavelength: float,
    focal_length: float,
    beam_radius_at_lens: float
) -> Tuple[float, float, float, float]:
    """
    Calculate all relevant beam parameters.
    
    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    focal_length : float
        Focal length [m]
    beam_radius_at_lens : float
        Beam radius at lens [m]
        
    Returns
    -------
    w0 : float
        Focused beam waist [m]
    zR : float
        Rayleigh range [m]
    k0 : float
        Wave number [rad/m]
    theta : float
        Divergence angle [rad]
    """
    w0 = calculate_focused_beam_waist(beam_radius_at_lens, wavelength, focal_length)
    zR = calculate_rayleigh_range(w0, wavelength)
    k0 = 2 * np.pi / wavelength
    theta = calculate_divergence_angle(w0, wavelength)
    
    return w0, zR, k0, theta

# ============================================================================
# PEAK INTENSITY CALCULATIONS
# ============================================================================

def calculate_peak_intensity(p_peak: float, w0: float) -> float:
    """
    Peak intensityof a Gaussian pulse.
    """
    if p_peak <= 0:
        raise ValueError(f"Peak power must be positive, got {p_peak}")
    if w0 <= 0:
        raise ValueError(f"w0 must be positive, got {w0}")

    A_eff = np.pi * w0**2 / 2.0
    return p_peak / A_eff

def calculate_peak_power(pulse_energy: float, pulse_duration: float) -> float:
    """
    Peak power of a Gaussian pulse.
    
    Parameters
    ----------
    pulse_energy : float
        Energy per pulse [J]
    pulse duration : float
        pulse duration [s]
        
    Returns
    -------
    float
        Peak power [W]
    """
    if pulse_energy <= 0:
        raise ValueError(f"Pulse energy must be positive, got {pulse_energy}")
    if pulse_duration <= 0:
        raise ValueError(f"Pulse duration must be positive, got {pulse_duration}")

    temporal_factor = np.sqrt(4 * np.log(2) / np.pi)
    return pulse_energy * temporal_factor / pulse_duration


def calculate_average_power(
    pulse_energy: float,
    repetition_rate: float
) -> float:
    """
    Calculate average power from pulse energy and repetition rate.
    
    Parameters
    ----------
    pulse_energy : float
        Energy per pulse [J]
    repetition_rate : float
        Pulse repetition rate [Hz]
        
    Returns
    -------
    float
        Average power [W]
    """
    if pulse_energy <= 0:
        raise ValueError(f"Pulse energy must be positive, got {pulse_energy}")
    if repetition_rate <= 0:
        raise ValueError(f"Repetition rate must be positive, got {repetition_rate}")
    
    return pulse_energy * repetition_rate


# ============================================================================
# ENERGY CALCULATIONS
# ============================================================================

def calculate_energy(field: np.ndarray) -> float:
    """
    Calculate the total energy (intensity integral) of a field.
    
    Parameters
    ----------
    field : np.ndarray
        Complex electric field
        
    Returns
    -------
    float
        Total energy
    """
    return float(np.sum(np.abs(field)**2))


def normalize_field(field: np.ndarray) -> np.ndarray:
    """
    Normalize a field to unit energy.
    
    Parameters
    ----------
    field : np.ndarray
        Complex electric field
        
    Returns
    -------
    np.ndarray
        Normalized field
    """
    E = calculate_energy(field)
    if E < _ENERGY_THRESHOLD:
        logger.warning("Field has near-zero energy, cannot normalize")
        return field
    return field / np.sqrt(E)