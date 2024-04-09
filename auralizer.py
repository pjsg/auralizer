"""
Usage:  
      ./auralizer.py  <sources>... [--freq <frequencies>]... [--out <outfile>]

Options:
      --freq <frequencies>                The center frequencies to use [default: 400, 900, 1900]
      --out <outfile>                     Destination wave file


Arguments:
      <sources>     The source files. These are the CSVs from the three radios.

"""

import numpy as np
import wave
import pandas as pd
import json
from docopt import docopt

scale_factor = 60 / 86400.0


def generate_sine_wave(
    freq, sample_rate, duration, start_phase=0, start_amplitude=1.0, end_amplitude=1.0
):
    """Generate a sine wave for a given frequency, sample rate, and duration, with phase and amplitude continuity."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.linspace(
        start_amplitude, end_amplitude, int(sample_rate * duration)
    )  # Linear amplitude interpolation
    phase = 2 * np.pi * freq * (t + 1 / sample_rate) + start_phase
    sine_wave = amplitude * np.sin(phase)
    end_phase = phase[-1] % (2 * np.pi)
    # print(f"t = {sample_rate * duration}, {sample_rate=} {duration=}: {phase=} {end_phase=}")
    return sine_wave, end_phase


def read_frequencies_and_vrms_from_csv(csv_file, base_freq, multiplier):
    """Read frequencies and VRMS values from a CSV file, applying frequency mapping and assuming rows indicate timing."""
    df = pd.read_csv(csv_file, comment="#")
    df["Start"] = df.index * scale_factor
    # Apply frequency mapping formula
    df["MappedFreq"] = (df["Freq"] - 5000000) * multiplier + base_freq
    frequencies_and_vrms = df[["Start", "MappedFreq", "Vrms"]].values.tolist()
    return frequencies_and_vrms


# Parameters

sample_rate = 8000
scale_factor = int(sample_rate * scale_factor) / sample_rate
bit_depth = 16


def import_file(csv_file_path, base_freq, multiplier):
    frequencies_and_vrms = read_frequencies_and_vrms_from_csv(csv_file_path, base_freq, multiplier)
    duration = frequencies_and_vrms[-1][0] + scale_factor

    # Initialize the signal
    signal = np.zeros(int(sample_rate * duration))
    current_phase = 0
    current_amplitude = frequencies_and_vrms[0][2] if frequencies_and_vrms else 1.0
    last_end_index = 0

    # Generate the signal
    for i, (start_time, mapped_freq, vrms) in enumerate(frequencies_and_vrms):
        next_amplitude = vrms if i + 1 < len(frequencies_and_vrms) else vrms
        segment_duration = scale_factor
        sine_wave_segment, current_phase = generate_sine_wave(
            mapped_freq,
            sample_rate,
            segment_duration,
            current_phase,
            current_amplitude,
            next_amplitude,
        )
        start_index = last_end_index
        end_index = start_index + len(sine_wave_segment)
        last_end_index = end_index
        signal[start_index:end_index] = sine_wave_segment
        current_amplitude = next_amplitude  # Prepare for the next segment

    max = np.percentile(signal, 98)
    return signal / max


if __name__ == "__main__":
    args = docopt(__doc__)
    # print(json.dumps(args))
    inputs = args["<sources>"]

    ch1 = 400
    ch2 = 900
    ch3 = 2100

    try:
        ch1 = int(args["--freq"][0])
        ch2 = int(args["--freq"][1])
        ch3 = int(args["--freq"][2])
    except Exception:
        pass

    signal = import_file(inputs[0], ch1, ch1 / 2.5)
    signal += import_file(inputs[1], ch2, ch2 / 2.5)
    signal += import_file(inputs[2], ch3, ch3 / 2.5)

    # Convert to 16-bit data
    max = np.max(signal)
    signal_normalized = np.int16(signal * 32767 * 0.9 / max)

    # Write to a WAV file
    file_path = args["--out"] or "auralizer_output.wav"
    with wave.open(file_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_normalized)
