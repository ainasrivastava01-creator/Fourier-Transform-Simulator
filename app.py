
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Fourier Transform Simulator",
    page_icon="📊",
    layout="wide"
)

st.title("Fourier Transform Simulator")
st.markdown(
    "### Interactive Signals and Systems Laboratory"
)
st.write(
    "Generate signals, calculate their Fourier Transform, "
    "analyze frequency components and reconstruct the signal."
)

# ----------------------------------
# SIDEBAR CONTROLS
# ----------------------------------

st.sidebar.header("Simulation Controls")

signal_type = st.sidebar.selectbox(
    "Select Input Signal",
    ["Sine Wave", "Cosine Wave", "Square Wave", "Composite Signal"]
)

amplitude = st.sidebar.slider(
    "Amplitude",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

frequency = st.sidebar.slider(
    "Frequency (Hz)",
    min_value=1,
    max_value=100,
    value=5
)

sampling_rate = st.sidebar.select_slider(
    "Sampling Frequency (Hz)",
    options=[50, 100, 200, 500, 1000, 2000, 5000],
    value=1000
)

duration = st.sidebar.slider(
    "Signal Duration (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

phase_degrees = st.sidebar.slider(
    "Phase Angle (degrees)",
    min_value=0,
    max_value=360,
    value=0
)

# ----------------------------------
# GENERATE INPUT SIGNAL
# ----------------------------------

N = int(sampling_rate * duration)

t = np.arange(N) / sampling_rate

phase = np.deg2rad(phase_degrees)

if signal_type == "Sine Wave":
    signal = amplitude * np.sin(
        2 * np.pi * frequency * t + phase
    )

elif signal_type == "Cosine Wave":
    signal = amplitude * np.cos(
        2 * np.pi * frequency * t + phase
    )

elif signal_type == "Square Wave":
    signal = amplitude * np.sign(
        np.sin(2 * np.pi * frequency * t + phase)
    )

else:
    signal = (
        amplitude * np.sin(2 * np.pi * frequency * t + phase)
        + 0.5 * amplitude * np.sin(
            2 * np.pi * (frequency * 3) * t
        )
    )

# ----------------------------------
# FOURIER TRANSFORM (FFT)
# ----------------------------------

fft_result = np.fft.rfft(signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

# Magnitude Spectrum
magnitude = np.abs(fft_result) / N

if N > 1:
    if N % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2

# Phase Spectrum
phase_spectrum = np.angle(fft_result, deg=True)

# ----------------------------------
# INVERSE FOURIER TRANSFORM
# ----------------------------------

reconstructed_signal = np.fft.irfft(
    fft_result,
    n=N
)

# ----------------------------------
# DISPLAY CALCULATED VALUES
# ----------------------------------

st.subheader("Simulation Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Signal Type", signal_type)
col2.metric("Sampling Rate", f"{sampling_rate} Hz")
col3.metric("Total Samples", N)
col4.metric("Frequency", f"{frequency} Hz")

st.divider()

# ----------------------------------
# PLOT GRAPHS
# ----------------------------------

st.subheader("Signal Analysis")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Graph 1: Original Signal

axes[0, 0].plot(t, signal, linewidth=1.5)
axes[0, 0].set_title("Original Input Signal")
axes[0, 0].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True)

# Graph 2: Magnitude Spectrum

axes[0, 1].plot(
    frequencies,
    magnitude,
    linewidth=1.5
)
axes[0, 1].set_title("Fourier Transform - Magnitude Spectrum")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].grid(True)

# Graph 3: Phase Spectrum

threshold = max(np.max(magnitude) * 0.01, 1e-10)
valid = magnitude > threshold

axes[1, 0].stem(
    frequencies[valid],
    phase_spectrum[valid],
    basefmt=" "
)
axes[1, 0].set_title("Phase Spectrum")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Phase (degrees)")
axes[1, 0].grid(True)

# Graph 4: Reconstructed Signal

axes[1, 1].plot(
    t,
    signal,
    label="Original Signal",
    linewidth=2
)

axes[1, 1].plot(
    t,
    reconstructed_signal,
    "--",
    label="Reconstructed Signal",
    linewidth=1.5
)

axes[1, 1].set_title("Inverse Fourier Transform")
axes[1, 1].set_xlabel("Time (seconds)")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ----------------------------------
# DOMINANT FREQUENCY DETECTION
# ----------------------------------

st.subheader("Frequency Components")

if len(magnitude) > 1:
    peaks, _ = find_peaks(magnitude)

    if len(peaks) == 0:
        peaks = np.array([np.argmax(magnitude[1:]) + 1])

    strongest = peaks[
        np.argsort(magnitude[peaks])[::-1]
    ][:5]

    st.write("Strongest detected frequency components:")

    for i, peak in enumerate(strongest, start=1):
        st.write(
            f"{i}. Frequency: {frequencies[peak]:.2f} Hz "
            f"| Magnitude: {magnitude[peak]:.3f}"
        )

# ----------------------------------
# ERROR CALCULATION
# ----------------------------------

error = np.max(
    np.abs(signal - reconstructed_signal)
)

st.metric(
    "Maximum Reconstruction Error",
    f"{error:.10f}"
)

# ----------------------------------
# THEORY SECTION
# ----------------------------------

with st.expander("Understand Fourier Transform"):
    st.write(
        "Fourier Transform converts a signal from "
        "the time domain into the frequency domain."
    )

    st.latex(
        r"X(f)=\int_{-\infty}^{\infty}"
        r"x(t)e^{-j2\pi ft}\,dt"
    )

    st.write(
        "FFT (Fast Fourier Transform) is an efficient "
        "algorithm to calculate the Discrete Fourier Transform."
    )

    st.latex(
        r"x[n]=\frac{1}{N}\sum_{k=0}^{N-1}"
        r"X[k]e^{j2\pi kn/N}"
    )

st.success("Simulation completed successfully!")
