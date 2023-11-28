import numpy as np
from scipy.io import wavfile

def pink_noise_weighting(frequencies):
    """
    Calculate pink noise weighting for given frequencies.
    Pink noise decreases in power by 3 dB per octave.
    """
    # Pink noise has a 1/f frequency response
    return 1 / np.sqrt(frequencies)

def generate_weighted_ofdm_channel(num_subcarriers, sample_rate, signal_length, freq_low, freq_high):
    """
    Generate an OFDM signal for one channel with pink noise weighting.
    """
    t = np.linspace(0, signal_length, int(sample_rate * signal_length), endpoint=False)
    signal = np.zeros_like(t, dtype=np.complex_)

    # Frequency step and range for subcarriers
    freq_step = (freq_high - freq_low) / num_subcarriers
    frequencies = np.arange(freq_low, freq_high, freq_step)

    # Apply pink noise weighting
    weights = pink_noise_weighting(frequencies)

    for k in range(num_subcarriers):
        freq = freq_low + k * freq_step
        phase = np.random.uniform(0, 2 * np.pi)
        weighted_amplitude = weights[k]
        signal += weighted_amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))

    return np.real(signal)

def generate_pink_noise_ofdm(n_channels, num_subcarriers, sample_rate, signal_length):
    """
    Generate N-channel OFDM stimuli with pink noise spectrum.
    """
    signals = []
    for i in range(n_channels):
        signal = generate_weighted_ofdm_channel(num_subcarriers, sample_rate, signal_length, 20, 20000)

        # Normalize to prevent clipping
        signal = signal / np.max(np.abs(signal))
        signals.append(signal)

    # Stack signals for multi-channel format
    return np.stack(signals, axis=-1)

# Define parameters
n_channels = 4
num_subcarriers = 20000  # Number of subcarriers for each channel
signal_length = 10.0   # Length of the signal in seconds
sample_rate = 48000    # Sample rate in Hz

# Generate the stimuli
pink_noise_stimuli = generate_pink_noise_ofdm(n_channels, num_subcarriers, sample_rate, signal_length)

# Save to WAV file
output_filename = "pink_noise_ofdm_stimuli-20000.wav"
wavfile.write(output_filename, sample_rate, pink_noise_stimuli.astype(np.float32))

output_filename
