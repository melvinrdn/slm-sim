import numpy as np
import slm_sim

logger = slm_sim.setup_logger()

logger.info("=" * 70)
logger.info("START OF THE SIMULATION")
logger.info("=" * 70)

# ============================================================================
# LASER PARAMETERS
# ============================================================================

WAVELENGTH = 1.03e-6
PULSE_ENERGY = 700e-6
PULSE_DURATION = 180e-15  # FWHM
REPETITION_RATE = 1e4

P_AVG = slm_sim.calculate_average_power(PULSE_ENERGY, REPETITION_RATE)
P_PEAK = slm_sim.calculate_peak_power(PULSE_ENERGY, PULSE_DURATION)

logger.info("")
logger.info("LASER PARAMETERS")
logger.info("=" * 70)
logger.info(f"Wavelength:           {WAVELENGTH*1e9:.1f} nm")
logger.info(f"Pulse energy:         {PULSE_ENERGY*1e6:.1f} µJ")
logger.info(f"Pulse duration:       {PULSE_DURATION*1e15:.0f} fs (FWHM)")
logger.info(f"Repetition rate:      {REPETITION_RATE*1e-3:.1f} kHz")
logger.info(f"Average power:        {P_AVG:.2f} W")
logger.info(f"Peak power:           {P_PEAK:.2e} W")
logger.info("=" * 70)

# ============================================================================
# SYSTEM
# ============================================================================

FOCAL_LENGTH = 0.175
BEAM_RADIUS_AT_LENS = 3.5e-3  # 1/e² radius

w0, zR, k0, theta = slm_sim.calculate_beam_parameters(
    WAVELENGTH, FOCAL_LENGTH, BEAM_RADIUS_AT_LENS
)

I_peak_theory = slm_sim.calculate_peak_intensity(P_PEAK, w0)

logger.info("")
logger.info("FOCUSED BEAM PARAMETERS")
logger.info("=" * 70)
logger.info(f"Focal length:         {FOCAL_LENGTH*1e3:.1f} mm")
logger.info(f"Focused beam waist:   {w0*1e6:.2f} µm")
logger.info(f"Rayleigh range:       {zR*1e3:.2f} mm")
logger.info(f"Divergence angle:     {np.rad2deg(theta):.3f}°")
logger.info(f"Peak intensity: {I_peak_theory/1e18:.2f} × 10e14 W/cm²")

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

SLM_SIZE = 15.0 * BEAM_RADIUS_AT_LENS
NUM_POINTS = 1024
QUANTIZE_BITS = 10

Z_RANGE = (-1, 1)
NUM_Z_STEPS = 50

logger.info("")
logger.info("SIMULATION PARAMETERS")
logger.info("=" * 70)
logger.info(f"SLM size:             {SLM_SIZE*1e3:.2f} mm")
logger.info(f"Grid points:          {NUM_POINTS}×{NUM_POINTS}")
logger.info(f"Sampling:             {SLM_SIZE*2e6/NUM_POINTS:.2f} µm/pixel")
logger.info(f"Phase quantization:   {2**QUANTIZE_BITS} levels ({QUANTIZE_BITS}-bit)")
logger.info(f"Z-scan range:         {Z_RANGE[0]} to {Z_RANGE[1]} zR")
logger.info(f"Z-scan steps:         {NUM_Z_STEPS}")

# ============================================================================
# CREATE SLM FIELD
# ============================================================================

logger.info("Creating SLM field...")

x_slm, y_slm, extent_slm = slm_sim.create_slm_coordinates(
    SLM_SIZE, NUM_POINTS, verbose=True
)

intensity_slm = slm_sim.create_intensity_gaussian(
    x_slm, y_slm, BEAM_RADIUS_AT_LENS, normalize=True
)

phase_slm = slm_sim.create_phase_vortex(x_slm,y_slm,charge=1)

phase_slm = slm_sim.quantize_phase(
    phase_slm,
    bits=QUANTIZE_BITS,
    phase_range=(0, 2*np.pi)
)

logger.info(f"Phase quantized to {2**QUANTIZE_BITS} levels")

E_slm = np.sqrt(intensity_slm) * np.exp(1j * phase_slm)
E_in = slm_sim.calculate_energy(E_slm)
logger.info(f"Input field created (energy: {E_in:.2e})")
logger.info("=" * 70)

# ============================================================================
# PROPAGATE TO FOCAL PLANE
# ============================================================================

logger.info("")
logger.info("PROPAGATION")
logger.info("=" * 70)

from slm_sim.utils.arrays import zero_padding_2N, clip_to_original_size

E_slm_padded = zero_padding_2N(E_slm)

E_focus_padded = slm_sim.propagate(
    E_slm_padded,
    method='fourier',
    wavelength=WAVELENGTH,
    L=SLM_SIZE * 2,
    direction="forward",
    workers=-1
)

E_focus = clip_to_original_size(E_focus_padded, E_slm)

E_out = slm_sim.calculate_energy(E_focus)
rel_error = abs(E_out - E_in) / E_in

logger.info(f"Output field energy:  {E_out:.2e}")
logger.info(f"Energy conservation:  {(1-rel_error)*100:.2f}%")

x_focus, y_focus, extent_focus = slm_sim.create_focal_plane_coordinates(
    SLM_SIZE, NUM_POINTS, FOCAL_LENGTH, WAVELENGTH, verbose=False
)
dx_focus = (x_focus[-1] - x_focus[0]) / (len(x_focus) - 1)
logger.info("=" * 70)

# ============================================================================
# Z-SCAN PROPAGATION
# ============================================================================

logger.info(f"Performing z-scan from {Z_RANGE[0]} to {Z_RANGE[1]} zR...")

fields_z, z, x_z, y_z = slm_sim.propagate_z_scan(
    E_focus,
    z_range=Z_RANGE,
    num_steps=NUM_Z_STEPS,
    wavelength=WAVELENGTH,
    rayleigh_range=zR,
    size_slm=SLM_SIZE,
    num_points_slm=NUM_POINTS,
    bandlimit=True,
    workers=-1,
    crop_margin=3,
    verbose=True
)

logger.info(f"Z-scan complete: {fields_z.shape[2]} planes")
logger.info(f"Output grid: {fields_z.shape[0]}×{fields_z.shape[1]} points")
logger.info(f"Field size: {(x_z[-1]-x_z[0])*1e3:.2f} mm")
logger.info("=" * 70)

# ============================================================================
# ANALYSIS
# ============================================================================

logger.info("")
logger.info("ANALYSIS AND PLOTS")
logger.info("=" * 70)

intensities_z = np.abs(fields_z)**2
max_intensity_vs_z = np.max(intensities_z, axis=(0, 1))
total_power_vs_z = np.sum(intensities_z, axis=(0, 1))
z_zero_index = np.argmin(np.abs(z))

I_peak_max = max_intensity_vs_z.max()
I_peak_min = max_intensity_vs_z.min()
logger.info(f"Peak intensity variation: {I_peak_max / I_peak_min:.2f}×")

power_variation = total_power_vs_z.max() / total_power_vs_z.min()
logger.info(f"Power variation along z:  {(power_variation-1)*100:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

logger.info("")
logger.info("Creating plots...")

fig1, axs1 = slm_sim.plot_slm_field(
    E_slm,
    extent_slm,
    slm_size=(15.36e-3, 15.36e-3),
    show_contour=False,
    save_path="fig/results_slm.png"
)

fig2, axs2 = slm_sim.plot_focal_field(
    fields_z[:, :, z_zero_index],
    [x_z[0], x_z[-1], y_z[0], y_z[-1]],
    window_um=(1000, 1000),
    show_beam_width=False,
    save_path="fig/results_focus.png"
)

fig3, axs3 = slm_sim.plot_intensity_profile(
    fields_z[:, :, z_zero_index],
    x_z,
    y_z,
    save_path="fig/results_profiles.png"
)

fig4, axs4 = slm_sim.plot_z_scan(
    fields_z, x_z, y_z, z,
    rayleigh_range=zR,
    window_um=(1000, 1000),
    show_beam_width=False,
    save_path="fig/results_zscan.png"
)

logger.info("=" * 70)
logger.info("SIMULATION COMPLETE")
logger.info("=" * 70)
