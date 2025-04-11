import streamlit as st
import numpy as np
import soundfile as sf
import io
import hashlib
import math

########################################
# Chaotic Encryption (Minimal Example)
########################################

def rossler_derivatives(state, a, b, c):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(state, dt, a, b, c):
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence(n, dt, a, b, c, x0, y0, z0, burn_in=200):
    """
    Generate a 1D chaotic sequence using the Rossler attractor's x-value,
    normalized to [0, 1].
    """
    state = np.array([x0, y0, z0], dtype=float)
    # Burn-in
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    arr = np.array(sequence)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)

def derive_initial_conditions(passphrase):
    """Generate chaotic initial conditions from a passphrase (SHA256)."""
    digest = hashlib.sha256(passphrase.encode()).hexdigest()
    norm = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(digest[0:21], 16) / norm
    y0 = int(digest[21:42], 16) / norm
    z0 = int(digest[42:64], 16) / norm
    return x0, y0, z0

def encrypt_audio(audio_samples, passphrase,
                  dt=0.01, a=0.2, b=0.2, c=5.7,
                  burn_in=200, chaos_mod_scale=1.0):
    """
    Example encryption:
      1. Generate chaotic sequence matching length of audio.
      2. Multiply audio by (1 + chaotic_offset*chaos_mod_scale).
    """
    x0, y0, z0 = derive_initial_conditions(passphrase)
    n = len(audio_samples)
    chaos_seq = generate_chaotic_sequence(n, dt, a, b, c, x0, y0, z0, burn_in=burn_in)

    # Apply a simple "modulation" for demonstration:
    # out[i] = in[i] * (1 + chaos_seq[i]*chaos_mod_scale)
    # (You can replace with your own oscilLOCK scheme if needed.)
    encrypted = audio_samples * (1.0 + chaos_seq * chaos_mod_scale)
    return encrypted

########################################
# Metric Calculations
########################################

def compute_entropy(signal, bins=256):
    """
    Compute Shannon entropy of the signal samples by
    mapping them to a 0..255 range.
    """
    # Normalize to 0..1 for binning
    normed = (signal - signal.min()) / (signal.ptp() + 1e-12)
    hist, _ = np.histogram(normed, bins=bins, range=(0,1), density=True)
    hist = hist[hist>0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def compute_correlation(original, encrypted):
    """
    Pearson correlation coefficient between two signals.
    We want it near zero for good encryption.
    """
    # If lengths differ, crop to the min
    length = min(len(original), len(encrypted))
    orig = original[:length]
    enc  = encrypted[:length]
    corr = np.corrcoef(orig, enc)[0, 1]
    return corr

def compute_npcr(original, encrypted):
    """
    NPCR (Number of Pixel Change Rate), adapted for 1D.
    Typically used for images, but we can treat each sample as a 'pixel'.
      NPCR = (# of changed samples / total samples) * 100%
    We'll define "changed" if the samples differ by > 1e-12 after normalizing 0..1
    """
    length = min(len(original), len(encrypted))
    orig = original[:length]
    enc  = encrypted[:length]
    # Convert to 0..255
    def to_255(s):
        s_norm = (s - s.min())/(s.ptp()+1e-12)
        return np.rint(s_norm*255).astype(np.int32)
    
    orig_255 = to_255(orig)
    enc_255  = to_255(enc)
    diff_count = np.sum(orig_255 != enc_255)
    npcr = (diff_count / float(length))*100
    return npcr

def compute_uaci(original, encrypted):
    """
    UACI (Unified Average Changing Intensity), adapted for 1D signals.
    Typically for images: UACI = (1/N)*sum(|C1 - C2| / 255) * 100%
    We'll do similar for 1D, normalizing to 0..255.
    """
    length = min(len(original), len(encrypted))
    orig = original[:length]
    enc  = encrypted[:length]
    
    def to_255(s):
        s_norm = (s - s.min())/(s.ptp()+1e-12)
        return np.rint(s_norm*255).astype(np.int32)
    
    orig_255 = to_255(orig)
    enc_255  = to_255(enc)
    diff = np.abs(orig_255 - enc_255)
    uaci = (np.mean(diff) / 255.0)*100
    return uaci

def estimate_key_space():
    """
    A placeholder to show an example key-space.
    If your passphrase or chaotic system truly has 128 bits or more,
    you can put that. Otherwise, adapt as needed.
    """
    return "2^128 (theoretical)"

########################################
# Streamlit App
########################################

def main():
    st.set_page_config(page_title="Chaos Audio Encryption Tester", layout="wide")
    st.title("Chaos‐Based Audio Encryption: Metric Evaluation")

    st.sidebar.header("Encryption Input")
    # User uploads an audio file
    uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV/FLAC/OGG)", type=["wav","flac","ogg"])
    
    # Passphrase for chaotic encryption
    passphrase = st.sidebar.text_input("Passphrase:", "DefaultPassphrase")
    
    # Show "Encrypt & Compute Metrics" button
    if st.sidebar.button("Encrypt & Compute Metrics"):
        if uploaded_file:
            try:
                # Read audio
                data, sr = sf.read(uploaded_file)
                # Convert to float32 if in int16
                if data.dtype == np.int16:
                    data = data.astype(np.float32)/32767.0
            except Exception as e:
                st.error(f"Error reading audio: {e}")
                return

            # 1) Original Audio
            # 2) Perform encryption
            encrypted = encrypt_audio(
                audio_samples=data,
                passphrase=passphrase,
                dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                burn_in=BURN_IN,
                chaos_mod_scale=1.0  # you can adjust scaling factor
            )

            # Compute metrics
            ent_orig = compute_entropy(data)
            ent_enc  = compute_entropy(encrypted)
            corr     = compute_correlation(data, encrypted)
            npcr_val = compute_npcr(data, encrypted)
            uaci_val = compute_uaci(data, encrypted)
            kspace   = estimate_key_space()
            
            # Display results
            st.subheader("Encryption Results & Metrics")

            st.write(f"**Sample Rate:** {sr} Hz")
            st.write(f"**Original Audio Length:** {len(data)} samples")
            
            # Entropy
            st.write(f"**Original Entropy:** {ent_orig:.4f}")
            st.write(f"**Encrypted Entropy:** {ent_enc:.4f}")
            
            # Correlation
            st.write(f"**Correlation (Plain vs Encrypted):** {corr:.4f}")
            
            # NPCR & UACI
            st.write(f"**NPCR:** {npcr_val:.2f}%")
            st.write(f"**UACI:** {uaci_val:.2f}%")

            # Key space
            st.write(f"**Key Space (theoretical):** {kspace}")
            
            # Provide encrypted audio for user to download if desired
            with io.BytesIO() as buf:
                sf.write(buf, encrypted, sr, format="WAV")
                audio_bytes = buf.getvalue()
            
            st.download_button("Download Encrypted Audio (WAV)",
                               data=audio_bytes,
                               file_name="encrypted_audio.wav",
                               mime="audio/wav")
            
        else:
            st.warning("Please upload an audio file first.")

if __name__ == "__main__":
    main()
