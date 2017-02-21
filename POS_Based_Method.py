
import numpy as np


def extract_pos_based_method_improved(_time_series, _fps):

    # sliding window size
    window_size = 48
    hann_window = np.hanning(window_size)
    overlap = 24
    signal_length = len(_time_series)-overlap
    H = np.zeros(signal_length, dtype='float64')

    windows_counter = int((len(_time_series) - window_size) / overlap)
    # windows_counter = int(signal_length / window_size) * 2

    n = window_size
    m = n - window_size
    for i in range(windows_counter):

        window = _time_series[m:n]

        hann_windowed_signal = rgb_into_pulse_signal(window, hann_window)

        # Overlap-adding
        H[m:n] += hann_windowed_signal
        n += overlap
        m = n - window_size

    # last window is splitted by half and added at the end and front of H
    last_hann_windowed_signal = rgb_into_pulse_signal(_time_series[m:n], hann_window)

    # 1st half added at the back
    H[-overlap:] += last_hann_windowed_signal[:overlap]
    # 2nd half added at the front
    H[0:overlap] += last_hann_windowed_signal[overlap:]

    bpm, pruned_fft, heart_rates, fft, raw = get_bpm(H, _fps)

    return bpm, pruned_fft


def get_bpm(_H, _fps):

    # Fourier Transform
    raw = np.fft.fft(_H, 512)
    L = int(len(raw) / 2 + 1)
    fft = np.abs(raw[:L])

    frequencies = np.linspace(0, _fps/2.0, L, endpoint=True)

    heart_rates = frequencies * 60.0

    # bandpass filter for pulse
    bandpassed_fft = fft.copy()
    bound_low = (np.abs(heart_rates - 40)).argmin()
    bound_high = (np.abs(heart_rates - 170)).argmin()

    # pruned_fft = bandpassed_fft[bound_low:bound_high]

    bandpassed_fft[:bound_low] = 0
    bandpassed_fft[bound_high:] = 0

    max_freq_pos = np.argmax(bandpassed_fft)

    bpm = heart_rates[max_freq_pos]

    return bpm, bandpassed_fft, heart_rates, fft, raw


def rgb_into_pulse_signal(_window):

    # 5 temporal normalization
    mean_window = np.average(_window, axis=0)
    norm_window = _window / mean_window


    # 6 projection
    S1 = np.dot(norm_window, [-1, 1, 0])
    S2 = np.dot(norm_window, [1, 1, -2])

    # 7 tuning
    S1_std = np.std(S1)
    S2_std = np.std(S2)

    alpha = S1_std / S2_std

    h = S1 + alpha * S2

    # # Hann window signal
    # _hann_windowed_signal = _hann_window * h

    return h, norm_window


