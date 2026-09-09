# Power Spectral Density Estimates Using FFT
This document demonstrates how to estimate power spectral density (PSD) directly from the FFT in Python. The approach follows the standard periodogram scaling and can be compared with the MATLAB example provided by [MathWorks](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html).

## Example
This example will use even-length input to simplify one-sided FFT calculations. See [FFT examples in Python and Matlab](fft_python_matlab.md) for more general code.

Create a sinusoidal signal with zero-mean Gaussian noise.

```python
import numpy as np

# Sampling frequency
Fs = 1000

# Time signal for 1 second (number of samples = sampling frequency)
t = np.arange(0, 1, 1/Fs)

# Sinusoidal signal with zero-mean Gaussian noise (use rng for reproducible noise)
x = np.cos(2*np.pi*100*t) + np.random.standard_normal(size=t.shape)

```

Compute periodogram using FFT
```python
# Number of samples
N = len(x)

# Compute the two-sided FFT
X = np.fft.fft(x)

# Get to one-sided FFT and compute corresponding frequencies
X = X[:N//2+1]
F = np.arange(N//2+1) * Fs/N

# Compute periodogram
Pxx = np.abs(X)**2 / (Fs*N)
Pxx[1:-1] *= 2

```

Plot the power in decibels (dB)
```python
from matplotlib import pyplot as plt

plt.figure()
plt.plot(F, 10*np.log10(Pxx))
plt.title("Periodogram Using FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(True)

```

Compute periodogram using scipy function `periodogram` and plot.
```python
from scipy import signal

# Compute PSD
# Note: periodogram uses detrend='constant' by default, i.e. it removes the signal mean.
# Therefore, we use detrend=False for comparable results with the FFT method.
F_den, Pxx_den = signal.periodogram(x, Fs, detrend=False)

plt.figure()
plt.plot(F_den, 10*np.log10(Pxx_den))
plt.xlabel('Frequency [Hz]')
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(True)

# Compute maximum absolute deviation between the FFT and periodogram methods
mx_dev = np.max(np.abs(Pxx-Pxx_den))
print(f"Maximum absolute deviation: {max_dev:g}")

```

## Example with windowing

This example illustrates the effect of windowing and Welch's method on PSD estimation.

Create a sinusoidal signal with one bin-centered (100 Hz) and one non-bin-centered (110.7 Hz) component, plus Gaussian noise.

In the example we compare
* PSD using rectangular window and Hann window to illustrate spectral leakage
* PSD using Hann window periodogram and Welch's method.

Welch's method divides the signal into shorter segments, typically overlapping, and computes a windowed periodogram for each segment, and averages the result for the segments.

```python
# Sampling frequency
Fs = 1000

# Time signal for 1 second (number of samples = sampling frequency)
t = np.arange(0, 1, 1/Fs)

# Sinusoidal signal (100 and 110.7 Hz) with zero-mean Gaussian noise (use rng for reproducible noise)
x = np.cos(2*np.pi*100*t)
x += np.cos(2*np.pi*110.7*t)
x += 0.5 * np.random.standard_normal(size=t.shape)

# PSD using a default rectwin/boxcar window of signal length
F_rect, Pxx_rect = signal.periodogram(x, Fs, detrend=False)

# Using hann window
F_hann, Pxx_hann = signal.periodogram(x, Fs, window="hann", detrend=False)

# PSD using Welch's method with 256-sample segments and 50% overlap
F_welch, Pxx_welch = signal.welch(
    x,
    Fs,
    window="hann",
    nperseg=256,
    noverlap=128,
    detrend=False
)

# Plots comparing rectangular and Hann window
fig, ax = plt.subplots(2, 2, sharey=True)
ax11 = ax[0,0]
ax12 = ax[0,1]
ax21 = ax[1,0]
ax22 = ax[1,1]

ax11.plot(F_rect, 10*np.log10(Pxx_rect), label="Rect. window")
ax11.plot(F_hann, 10*np.log10(Pxx_hann), label="Hann window")

ax12.plot(F_rect, 10*np.log10(Pxx_rect), label="Rect. window")
ax12.plot(F_hann, 10*np.log10(Pxx_hann), label="Hann window")
ax12.set_xlim(80, 130)

# Plots comparing Hann window and Welch method
ax21.plot(F_hann, 10*np.log10(Pxx_hann), label="Hann window")
ax21.plot(F_welch, 10*np.log10(Pxx_welch), label="Welch method")

ax22.plot(F_hann, 10*np.log10(Pxx_hann), label="Hann window")
ax22.plot(F_welch, 10*np.log10(Pxx_welch), label="Welch method")
ax22.set_xlim(80, 130)

# Format axes
for a in ax.flat:
    a.grid(True)
    a.legend()

# Common axis labels
fig.supxlabel("Frequency (Hz)")
fig.supylabel("PSD (dB/Hz)")

plt.tight_layout()

```