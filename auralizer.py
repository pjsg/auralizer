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
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    amplitude = np.linspace(
        start_amplitude, end_amplitude, samples, endpoint=False
    )  # Linear amplitude interpolation
    phase = 2 * np.pi * freq * t + start_phase
    sine_wave = amplitude * np.sin(phase)
    end_phase = (2 * np.pi * freq * duration + start_phase) % (2 * np.pi)
    # print(f"t = {sample_rate * duration}, {sample_rate=} {duration=}: {phase=} {end_phase=}")
    return sine_wave, end_phase, end_amplitude


def read_frequencies_and_vrms_from_csv(csv_file, base_freq, octave_hz):
    """Read frequencies and VRMS values from a CSV file, applying frequency mapping and assuming rows indicate timing."""
    df = pd.read_csv(csv_file, comment="#")
    df["Start"] = df.index * scale_factor
    # Apply frequency mapping formula
    df["MappedFreq"] = 2 ** ((((df["Freq"] + 500000) % 1000000) - 500000) / octave_hz) * base_freq
    frequencies_and_vrms = df[["Start", "MappedFreq", "Vrms"]].values.tolist()
    return frequencies_and_vrms


# Parameters

sample_rate = 16000
scale_factor = int(sample_rate * scale_factor) / sample_rate  # Seconds per sample
bit_depth = 16


def import_file(csv_file_path, base_freq, octave_hz):
    frequencies_and_vrms = read_frequencies_and_vrms_from_csv(csv_file_path, base_freq, octave_hz)
    duration = frequencies_and_vrms[-1][0] + scale_factor

    # Initialize the signal
    signal = np.zeros(int(sample_rate * duration + 0.5))
    current_phase = 0
    current_amplitude = frequencies_and_vrms[0][2] if frequencies_and_vrms else 1.0
    last_end_index = 0

    # Generate the signal
    for i, (start_time, mapped_freq, vrms) in enumerate(frequencies_and_vrms):
        next_amplitude = vrms if i + 1 < len(frequencies_and_vrms) else vrms
        sine_wave_segment, current_phase, current_amplitude = generate_sine_wave(
            mapped_freq,
            sample_rate,
            scale_factor,
            current_phase,
            current_amplitude,
            next_amplitude,
        )
        start_index = last_end_index
        end_index = start_index + len(sine_wave_segment)
        last_end_index = end_index
        signal[start_index:end_index] = sine_wave_segment

    max = np.percentile(signal, 98)
    if max:
        return signal / max
    return signal


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

    df_stereo = pd.DataFrame(columns=["left", "right"])

    df_stereo["left"] = import_file(inputs[0], ch1, 2.0)
    df_stereo["right"] = import_file(inputs[2], ch3, 2.0)
    center = import_file(inputs[1], ch2, 2.0)
    df_stereo["left"] += center / 2
    df_stereo["right"] += center / 2

    # Convert to 16-bit data
    maxl = df_stereo["left"].max()
    maxr = df_stereo["right"].max()
    minl = df_stereo["left"].min()
    minr = df_stereo["right"].min()
    if not maxl:
        maxl = 1
    else:
        maxl = maxl if maxl > -minl else -minl
    if not maxr:
        maxr = 1
    else:
        maxr = maxr if maxr > -minr else -minr
    df_stereo["left"] = np.int16(df_stereo["left"] * (2 ** (bit_depth - 1)) * 0.9 / maxl)
    df_stereo["right"] = np.int16(df_stereo["right"] * (2 ** (bit_depth - 1)) * 0.9 / maxr)

    # Write to a WAV file
    file_path = args["--out"] or "auralizer_output.wav"
    with wave.open(file_path, "w") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        # Prepare data for writing (assuming little-endian byte order)
        frames = []
        for i in range(len(df_stereo)):
            left_sample = df_stereo.loc[i, "left"].tobytes()
            right_sample = df_stereo.loc[i, "right"].tobytes()
            frame = left_sample + right_sample
            frames.append(frame)

        # Write all frames to the wave file
        wav_file.writeframes(b"".join(frames))
