import streamlit as st
import numpy as np
import hashlib
import io
import os
from datetime import datetime
import scipy.io.wavfile as wav  # for reading WAV files

# ------------------------------------------------------------------
# Hard‑Coded Parameters (matching the oscilLOCK encryption settings)
# ------------------------------------------------------------------
TONE_DURATION    = 0.11        # seconds
GAP_DURATION     = 0.02        # seconds
BASE_FREQ        = 500         # Hz
FREQ_RANGE       = 1000        # Hz
CHAOS_MOD_RANGE  = 349.39      # Hz

NUM_CHAOTIC_SAMPLES = 704
BURN_IN         = 900

# Chaotic system parameters from grid search:
DT      = 0.005251616433272467   # seconds
A_PARAM = 0.12477067210511437
B_PARAM = 0.2852679643352883
C_PARAM = 6.801715623942842

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------
def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a space‑separated binary string back to text."""
    byte_list = binary_str.split()
    try:
        byte_array = bytearray(int(b, 2) for b in byte_list)
        return byte_array.decode(encoding)
    except Exception as e:
        return f"Decoding error: {e}"

def derive_initial_conditions(passphrase):
    """Derive initial conditions from the SHA‑256 hash of the passphrase."""
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()
    norm_const = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(hash_digest[0:21], 16) / norm_const
    y0 = int(hash_digest[21:42], 16) / norm_const
    z0 = int(hash_digest[42:64], 16) / norm_const
    return x0, y0, z0

# ------------------------------------------------------------------
# Chaotic System Functions (Rossler)
# ------------------------------------------------------------------
def rossler_derivatives(state, a, b, c):
    """Compute the derivatives for the Rossler attractor."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(state, dt, a, b, c):
    """Perform a single RK4 integration step for the Rossler system."""
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence_rossler_rk4(n, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM, 
                                          x0=0.1, y0=0.0, z0=0.0, burn_in=BURN_IN):
    """
    Generate a sequence of chaotic x-values using the Rossler attractor via RK4.
    The sequence is normalized to [0, 1].
    """
    state = np.array([x0, y0, z0], dtype=float)
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------------------------------------------------------
# Decryption Function
# ------------------------------------------------------------------
def decrypt_waveform_to_binary(waveform, sample_rate, tone_duration, gap_duration,
                               base_freq, freq_range, chaos_mod_range,
                               dt, a, b, c, passphrase):
    """
    Decrypt the provided audio waveform (encrypted via oscilLOCK) to recover a binary string.
    Steps:
      1. Segment the waveform into tone portions.
      2. Estimate the tone frequency via FFT and parabolic interpolation.
      3. Regenerate the chaotic sequence from the passphrase.
      4. Remove the chaotic modulation and invert the mapping to recover bytes.
    """
    tone_samples = int(sample_rate * tone_duration)
    gap_samples = int(sample_rate * gap_duration)
    segment_length = tone_samples + gap_samples
    total_samples = len(waveform)
    n_segments = total_samples // segment_length

    # Regenerate chaotic sequence using derived initial conditions
    x0, y0, z0 = derive_initial_conditions(passphrase)
    chaotic_sequence = generate_chaotic_sequence_rossler_rk4(n_segments, dt=dt, a=a, b=b, c=c,
                                                             x0=x0, y0=y0, z0=z0)
    binary_list = []
    for i in range(n_segments):
        start = i * segment_length
        end = start + tone_samples
        tone_segment = waveform[start:end]

        # Apply Hann window to tone segment
        N_tone = len(tone_segment)
        window = np.hanning(N_tone)
        windowed_tone = tone_segment * window

        # Zero-pad for increased FFT resolution (e.g., 4x the length)
        n_fft = int(2**np.ceil(np.log2(N_tone)) * 4)
        fft_result = np.fft.rfft(windowed_tone, n=n_fft)
        fft_magnitude = np.abs(fft_result)

        # Find the FFT peak
        peak_index = np.argmax(fft_magnitude)
        if 0 < peak_index < len(fft_magnitude)-1:
            alpha = fft_magnitude[peak_index-1]
            beta = fft_magnitude[peak_index]
            gamma = fft_magnitude[peak_index+1]
            p = 0.5*(alpha-gamma)/(alpha-2*beta+gamma)
        else:
            p = 0
        peak_index_adjusted = peak_index + p
        freq_resolution = sample_rate / n_fft
        observed_freq = peak_index_adjusted * freq_resolution

        # Remove chaotic modulation
        chaotic_offset = chaotic_sequence[i] * chaos_mod_range
        plain_freq = observed_freq - chaotic_offset

        # Invert mapping: plain_freq = base_freq + (byte_value/255)*freq_range
        byte_value = (plain_freq - base_freq) / freq_range * 255
        byte_value = int(np.rint(byte_value))
        byte_value = max(0, min(255, byte_value))
        binary_byte = format(byte_value, '08b')
        binary_list.append(binary_byte)
    
    binary_string = " ".join(binary_list)
    return binary_string

# ------------------------------------------------------------------
# Streamlit UI (Decryption Only; No Visualization)
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="oscilKEY - Decryption", layout="wide")
    st.title("oscilKEY: Audio Waveform Decryption")
    st.markdown("This app decrypts an encrypted WAV audio file (produced by oscilLOCK) to recover the original text message.")
    
    st.sidebar.header("Decryption Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Encrypted Audio (WAV)", type=["wav"])
    passphrase = st.sidebar.text_input("Enter Passphrase:", type="password", value="DefaultPassphrase")
    enter_button = st.sidebar.button("Enter")
    
    if uploaded_file and passphrase and enter_button:
        try:
            sr_file, waveform = wav.read(uploaded_file)
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32767.0
        except Exception as e:
            st.error(f"Error reading audio file: {e}")
            return
        
        with st.spinner("Decrypting..."):
            binary_output = decrypt_waveform_to_binary(
                waveform, sr_file,
                tone_duration=TONE_DURATION, gap_duration=GAP_DURATION,
                base_freq=BASE_FREQ, freq_range=FREQ_RANGE, chaos_mod_range=CHAOS_MOD_RANGE,
                dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                passphrase=passphrase
            )
            recovered_text = binary_to_text(binary_output)
        
        st.subheader("Decryption Output")
        st.markdown("**Recovered Binary:**")
        st.code(binary_output)
        st.subheader("Recovered Text:")
        st.write(recovered_text)

if __name__ == "__main__":
    main()
