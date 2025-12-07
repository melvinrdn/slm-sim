import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

_DEFAULT_PHASE_THRESHOLD = 1e-3
_DEFAULT_DPI = 300
_DEFAULT_INTENSITY_CMAP = 'turbo'
_DEFAULT_PHASE_CMAP = 'hsv'
_CONTOUR_LEVEL = 1 / np.e**2
_BEAM_WIDTH_COLOR = 'white'
_BEAM_WIDTH_LINEWIDTH = 2.0
_BEAM_WIDTH_LINESTYLE = '--'
_BEAM_WIDTH_ALPHA = 0.8


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def _validate_field(field: np.ndarray, name: str = "field") -> None:
    """Validate that field is a 2D array."""
    if not isinstance(field, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {field.shape}")


def _validate_coordinates(x: np.ndarray, y: np.ndarray) -> None:
    """Validate coordinate arrays."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Coordinates must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Coordinates must be 1D arrays")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _mask_phase_by_intensity(
    phase: np.ndarray,
    intensity: np.ndarray,
    threshold: float = _DEFAULT_PHASE_THRESHOLD,
    relative: bool = True
) -> np.ma.MaskedArray:
    """
    Mask phase values where intensity is below threshold.
    
    This prevents displaying noisy phase in low-intensity regions.
    """
    if relative:
        I_max = float(np.max(intensity)) if intensity.size else 0.0
        threshold_abs = threshold * I_max
    else:
        threshold_abs = threshold
    
    mask = intensity <= threshold_abs
    return np.ma.array(phase, mask=mask)


def _add_colorbar(
    fig: Figure,
    ax: Axes,
    im,
    size: str = "3.5%",
    pad: float = 0.08,
    **kwargs
) -> plt.Axes:
    """Add a colorbar to an axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    fig.colorbar(im, cax=cax, **kwargs)
    return cax


def _save_figure(
    fig: Figure,
    save_path: Optional[Union[str, Path]],
    dpi: int = _DEFAULT_DPI
) -> None:
    """Save figure to file if path is provided."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")


def _calculate_beam_widths(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate beam widths using second-order moments.
    
    Returns
    -------
    w_x, w_y : float
        Beam radius (2σ) in x and y directions [m]
    """
    from ..utils.arrays import calculate_second_order_moments
    
    intensity = np.abs(field)**2
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    
    d_x, d_y = calculate_second_order_moments(intensity, dx, use_center_of_mass=True)
    
    # Convert 4σ diameter to 2σ radius
    w_x = d_x / 2.0
    w_y = d_y / 2.0
    
    return w_x, w_y


def _overlay_beam_width_focal(
    ax: Axes,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    scale: float = 1e6,  # Convert to µm
    color: str = _BEAM_WIDTH_COLOR,
    linewidth: float = _BEAM_WIDTH_LINEWIDTH,
    linestyle: str = _BEAM_WIDTH_LINESTYLE,
    alpha: float = _BEAM_WIDTH_ALPHA
) -> None:
    """
    Overlay beam width contours on focal plane plot.
    """
    w_x, w_y = _calculate_beam_widths(field, x, y)
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_ellipse = w_x * np.cos(theta) * scale
    y_ellipse = w_y * np.sin(theta) * scale
    
    ax.plot(x_ellipse, y_ellipse, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha, label='Beam width (2σ)')


def _calculate_beam_widths_vs_z(
    fields_z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate beam widths for all z-planes.
    
    Returns
    -------
    w_x_vs_z, w_y_vs_z : np.ndarray
        Beam radii (2σ) vs z [m]
    """
    from ..utils.arrays import calculate_second_order_moments
    
    Ny, Nx, Nz = fields_z.shape
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    
    w_x_vs_z = np.zeros(Nz)
    w_y_vs_z = np.zeros(Nz)
    
    for iz in range(Nz):
        intensity = np.abs(fields_z[:, :, iz])**2
        d_x, d_y = calculate_second_order_moments(intensity, dx, use_center_of_mass=True)
        w_x_vs_z[iz] = d_x / 2.0  # Convert 4σ to 2σ
        w_y_vs_z[iz] = d_y / 2.0
    
    return w_x_vs_z, w_y_vs_z


def _overlay_beam_width_zscan(
    ax: Axes,
    fields_z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    direction: str,
    scale_spatial: float = 1e6,  # µm
    scale_z: float = 1.0,  # normalized to zR
    color: str = _BEAM_WIDTH_COLOR,
    linewidth: float = _BEAM_WIDTH_LINEWIDTH,
    linestyle: str = _BEAM_WIDTH_LINESTYLE,
    alpha: float = _BEAM_WIDTH_ALPHA
) -> None:
    """
    Overlay beam width evolution on z-scan plot.
    
    Parameters
    ----------
    direction : str
        'x' for x-z slice, 'y' for y-z slice
    """
    w_x_vs_z, w_y_vs_z = _calculate_beam_widths_vs_z(fields_z, x, y)
    
    if direction == 'x':
        w_vs_z = w_x_vs_z
    else:
        w_vs_z = w_y_vs_z
    
    # Plot both +w and -w
    ax.plot(w_vs_z * scale_spatial, z * scale_z, color=color,
            linewidth=linewidth, linestyle=linestyle, alpha=alpha,
            label='Beam width (2σ)')
    ax.plot(-w_vs_z * scale_spatial, z * scale_z, color=color,
            linewidth=linewidth, linestyle=linestyle, alpha=alpha)


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_slm_field(
    field: np.ndarray,
    extent: List[float],
    slm_size: Optional[Tuple[float, float]] = None,
    show_contour: bool = False,
    phase_threshold: float = 1e-6,
    figsize: Tuple[float, float] = (12.0, 4.8),
    intensity_cmap: str = _DEFAULT_INTENSITY_CMAP,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI
) -> Tuple[Figure, np.ndarray]:
    """
    Plot SLM plane field (intensity and phase).
    
    Parameters
    ----------
    field : np.ndarray
        2D complex field
    extent : list of float
        [x_min, x_max, y_min, y_max] in meters
    slm_size : tuple of float, optional
        (width, height) of SLM in meters (for zoom)
    show_contour : bool, optional
        If True, show 1/e² intensity contour
    phase_threshold : float, optional
        Mask phase below this relative intensity
    figsize : tuple of float, optional
        Figure size in inches
    intensity_cmap : str, optional
        Colormap for intensity
    save_path : str or Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    axs : np.ndarray
        Array of axes
        
    Raises
    ------
    ValueError
        If field is not 2D or extent has wrong length
    """
    _validate_field(field, "field")
    
    if len(extent) != 4:
        raise ValueError(f"extent must have 4 elements, got {len(extent)}")
    
    extent_mm = [v * 1e3 for v in extent]
    intensity = np.abs(field)**2
    I_max = float(np.max(intensity)) if intensity.size else 1.0
    phase = np.angle(field)
    phase_masked = _mask_phase_by_intensity(
        phase, intensity, threshold=phase_threshold, relative=True
    )
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("SLM Plane — Far Field", y=0.98)
    
    # Intensity plot
    im0 = axs[0].imshow(
        intensity / I_max, extent=extent_mm, origin='lower',
        aspect='equal', cmap=intensity_cmap
    )
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    axs[0].set_title("Normalized Intensity")
    
    if show_contour and I_max > 0:
        Ny, Nx = intensity.shape
        x_mm = np.linspace(extent_mm[0], extent_mm[1], Nx)
        y_mm = np.linspace(extent_mm[2], extent_mm[3], Ny)
        X_mm, Y_mm = np.meshgrid(x_mm, y_mm, indexing='xy')
        axs[0].contour(
            X_mm, Y_mm, intensity, levels=[I_max * _CONTOUR_LEVEL],
            colors='black', linestyles='--', linewidths=1.5
        )
    
    if slm_size is not None:
        w_mm, h_mm = slm_size[0] * 1e3, slm_size[1] * 1e3
        axs[0].set_xlim(-w_mm/2, w_mm/2)
        axs[0].set_ylim(-h_mm/2, h_mm/2)
    
    _add_colorbar(fig, axs[0], im0)
    
    # Phase plot
    im1 = axs[1].imshow(
        phase_masked, extent=extent_mm, origin='lower',
        aspect='equal', cmap=_DEFAULT_PHASE_CMAP, vmin=-np.pi, vmax=np.pi
    )
    axs[1].set_xlabel("x [mm]")
    axs[1].set_title("Phase")
    
    if show_contour and I_max > 0:
        axs[1].contour(
            X_mm, Y_mm, intensity, levels=[I_max * _CONTOUR_LEVEL],
            colors='black', linestyles='--', linewidths=1.5
        )
    
    if slm_size is not None:
        axs[1].set_xlim(-w_mm/2, w_mm/2)
        axs[1].set_ylim(-h_mm/2, h_mm/2)
    
    cax = _add_colorbar(fig, axs[1], im1)
    cax.set_yticks([-np.pi, 0, np.pi])
    cax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    
    plt.tight_layout()
    _save_figure(fig, save_path, dpi)
    
    return fig, axs


def plot_focal_field(
    field: np.ndarray,
    extent: List[float],
    window_um: Optional[Tuple[float, float]] = None,
    phase_threshold: float = _DEFAULT_PHASE_THRESHOLD,
    show_beam_width: bool = False,
    figsize: Tuple[float, float] = (12.0, 4.8),
    intensity_cmap: str = _DEFAULT_INTENSITY_CMAP,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI
) -> Tuple[Figure, np.ndarray]:
    """
    Plot focal plane field with optional beam width overlay.
    
    Parameters
    ----------
    field : np.ndarray
        2D complex field
    extent : list of float
        [x_min, x_max, y_min, y_max] in meters
    window_um : tuple of float, optional
        (width, height) display window in micrometers
    phase_threshold : float, optional
        Mask phase below this relative intensity
    show_beam_width : bool, optional
        If True, overlay 2σ beam width contour (second-order moments)
    figsize : tuple of float, optional
        Figure size in inches
    intensity_cmap : str, optional
        Colormap for intensity
    save_path : str or Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    axs : np.ndarray
        Array of axes
    """
    _validate_field(field, "field")
    
    if len(extent) != 4:
        raise ValueError(f"extent must have 4 elements, got {len(extent)}")
    
    extent_um = [v * 1e6 for v in extent]
    intensity = np.abs(field)**2
    I_max = float(np.max(intensity)) if intensity.size else 1.0
    phase = np.angle(field)
    phase_masked = _mask_phase_by_intensity(
        phase, intensity, threshold=phase_threshold, relative=True
    )
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Focal Plane — Near Field", y=0.98)
    
    # Intensity plot
    im0 = axs[0].imshow(
        intensity / I_max, extent=extent_um, origin='lower',
        aspect='equal', cmap=intensity_cmap
    )
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel("y [µm]")
    axs[0].set_title("Normalized Intensity")
    
    if window_um is not None:
        wx, wy = window_um
        axs[0].set_xlim(-wx/2, wx/2)
        axs[0].set_ylim(-wy/2, wy/2)
    
    # Overlay beam width if requested
    if show_beam_width:
        # Reconstruct coordinate arrays from extent
        Ny, Nx = field.shape
        x = np.linspace(extent[0], extent[1], Nx)
        y = np.linspace(extent[2], extent[3], Ny)
        _overlay_beam_width_focal(axs[0], field, x, y, scale=1e6)
        axs[0].legend(loc='upper right', framealpha=0.8)
    
    _add_colorbar(fig, axs[0], im0)
    
    # Phase plot
    im1 = axs[1].imshow(
        phase_masked, extent=extent_um, origin='lower',
        aspect='equal', cmap=_DEFAULT_PHASE_CMAP, vmin=-np.pi, vmax=np.pi
    )
    axs[1].set_xlabel("x [µm]")
    axs[1].set_title("Phase")
    
    if window_um is not None:
        wx, wy = window_um
        axs[1].set_xlim(-wx/2, wx/2)
        axs[1].set_ylim(-wy/2, wy/2)
    
    cax = _add_colorbar(fig, axs[1], im1)
    cax.set_yticks([-np.pi, 0, np.pi])
    cax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    
    plt.tight_layout()
    _save_figure(fig, save_path, dpi)
    
    return fig, axs


def plot_z_scan(
    fields_z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rayleigh_range: float,
    window_um: Optional[Tuple[float, float]] = None,
    phase_threshold: float = _DEFAULT_PHASE_THRESHOLD,
    show_beam_width: bool = False,
    figsize: Tuple[float, float] = (14.0, 8.0),
    intensity_cmap: str = _DEFAULT_INTENSITY_CMAP,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI
) -> Tuple[Figure, np.ndarray]:
    """
    Plot z-scan with optional beam width evolution overlay.
    
    Parameters
    ----------
    fields_z : np.ndarray
        3D array of shape (Ny, Nx, Nz) containing fields at each z-plane
    x, y : np.ndarray
        1D coordinate arrays in meters
    z : np.ndarray
        1D array of z-positions in meters
    rayleigh_range : float
        Rayleigh range in meters (for normalization)
    window_um : tuple of float, optional
        (width, height) display window in micrometers
    phase_threshold : float, optional
        Mask phase below this relative intensity
    show_beam_width : bool, optional
        If True, overlay 2σ beam width evolution (second-order moments)
    figsize : tuple of float, optional
        Figure size in inches
    intensity_cmap : str, optional
        Colormap for intensity
    save_path : str or Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    axs : np.ndarray
        2x2 array of axes
    """
    if fields_z.ndim != 3:
        raise ValueError(f"fields_z must be 3D, got shape {fields_z.shape}")
    _validate_coordinates(x, y)
    
    Ny, Nx, Nz = fields_z.shape
    y_center = Ny // 2
    x_center = Nx // 2
    
    x_um = x * 1e6
    y_um = y * 1e6
    z_norm = z / rayleigh_range
    
    # Extract x-z and y-z slices
    field_xz = fields_z[y_center, :, :].T
    field_yz = fields_z[:, x_center, :].T
    
    I_xz = np.abs(field_xz)**2
    I_yz = np.abs(field_yz)**2
    I_max = max(float(np.max(I_xz)), float(np.max(I_yz)))
    
    P_xz = np.angle(field_xz)
    P_yz = np.angle(field_yz)
    
    P_xz_masked = _mask_phase_by_intensity(P_xz, I_xz, phase_threshold, relative=True)
    P_yz_masked = _mask_phase_by_intensity(P_yz, I_yz, phase_threshold, relative=True)
    
    extent_xz = [x_um[0], x_um[-1], z_norm[0], z_norm[-1]]
    extent_yz = [y_um[0], y_um[-1], z_norm[0], z_norm[-1]]
    
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle("Propagation Scan", y=0.98)
    
    # X-Z Intensity
    im00 = axs[0, 0].imshow(
        I_xz / I_max, extent=extent_xz, origin='lower',
        aspect='auto', cmap=intensity_cmap
    )
    axs[0, 0].set_xlabel("x [µm]")
    axs[0, 0].set_ylabel(r"z [$z_R$]")
    axs[0, 0].set_title("Intensity x-z")
    if window_um is not None:
        axs[0, 0].set_xlim(-window_um[0]/2, window_um[0]/2)
    
    # Overlay beam width evolution
    if show_beam_width:
        _overlay_beam_width_zscan(
            axs[0, 0], fields_z, x, y, z, direction='x',
            scale_spatial=1e6, scale_z=1.0/rayleigh_range
        )
        axs[0, 0].legend(loc='upper right', framealpha=0.8, fontsize=8)
    
    _add_colorbar(fig, axs[0, 0], im00)
    
    # X-Z Phase
    im01 = axs[0, 1].imshow(
        P_xz_masked, extent=extent_xz, origin='lower',
        aspect='auto', cmap=_DEFAULT_PHASE_CMAP, vmin=-np.pi, vmax=np.pi
    )
    axs[0, 1].set_xlabel("x [µm]")
    axs[0, 1].set_ylabel(r"z [$z_R$]")
    axs[0, 1].set_title("Phase x-z")
    if window_um is not None:
        axs[0, 1].set_xlim(-window_um[0]/2, window_um[0]/2)
    cax01 = _add_colorbar(fig, axs[0, 1], im01)
    cax01.set_yticks([-np.pi, 0, np.pi])
    cax01.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    
    # Y-Z Intensity
    im10 = axs[1, 0].imshow(
        I_yz / I_max, extent=extent_yz, origin='lower',
        aspect='auto', cmap=intensity_cmap
    )
    axs[1, 0].set_xlabel("y [µm]")
    axs[1, 0].set_ylabel(r"z [$z_R$]")
    axs[1, 0].set_title("Intensity y-z")
    if window_um is not None:
        axs[1, 0].set_xlim(-window_um[1]/2, window_um[1]/2)
    
    # Overlay beam width evolution
    if show_beam_width:
        _overlay_beam_width_zscan(
            axs[1, 0], fields_z, x, y, z, direction='y',
            scale_spatial=1e6, scale_z=1.0/rayleigh_range
        )
        axs[1, 0].legend(loc='upper right', framealpha=0.8, fontsize=8)
    
    _add_colorbar(fig, axs[1, 0], im10)
    
    # Y-Z Phase
    im11 = axs[1, 1].imshow(
        P_yz_masked, extent=extent_yz, origin='lower',
        aspect='auto', cmap=_DEFAULT_PHASE_CMAP, vmin=-np.pi, vmax=np.pi
    )
    axs[1, 1].set_xlabel("y [µm]")
    axs[1, 1].set_ylabel(r"z [$z_R$]")
    axs[1, 1].set_title("Phase y-z")
    if window_um is not None:
        axs[1, 1].set_xlim(-window_um[1]/2, window_um[1]/2)
    cax11 = _add_colorbar(fig, axs[1, 1], im11)
    cax11.set_yticks([-np.pi, 0, np.pi])
    cax11.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    
    _save_figure(fig, save_path, dpi)
    
    return fig, axs


def plot_intensity_profile(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
    figsize: Tuple[float, float] = (10.0, 4.0),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI
) -> Tuple[Figure, np.ndarray]:
    """
    Plot horizontal and vertical intensity profiles through center.
    
    Parameters
    ----------
    field : np.ndarray
        2D complex field
    x, y : np.ndarray
        1D coordinate arrays in meters
    normalize : bool, optional
        If True, normalize to maximum intensity
    figsize : tuple of float, optional
        Figure size in inches
    save_path : str or Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    axs : np.ndarray
        Array of axes
    """
    _validate_field(field, "field")
    _validate_coordinates(x, y)
    
    intensity = np.abs(field)**2
    Ny, Nx = intensity.shape
    y_center = Ny // 2
    x_center = Nx // 2
    
    I_x = intensity[y_center, :]
    I_y = intensity[:, x_center]
    
    if normalize:
        I_max = max(float(np.max(I_x)), float(np.max(I_y)))
        if I_max > 0:
            I_x = I_x / I_max
            I_y = I_y / I_max
        ylabel = "Normalized Intensity"
    else:
        ylabel = "Intensity [a.u.]"
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    axs[0].plot(x * 1e6, I_x, 'b-', linewidth=2)
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title("Horizontal Profile")
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(y * 1e6, I_y, 'r-', linewidth=2)
    axs[1].set_xlabel("y [µm]")
    axs[1].set_ylabel(ylabel)
    axs[1].set_title("Vertical Profile")
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, save_path, dpi)
    
    return fig, axs