"""A small spoken-digit baseline; no EEG or conversational inference."""

from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import spectrogram
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
COMMIT = "d6938f9bf1545aa66d8489fc9f1385a7abd64282"
URL = f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{COMMIT}"
SEED = 42


def download():
    """Fetch one fixed upstream revision, retaining its attribution."""
    if (DATA / "manifest.json").exists():
        return
    DATA.mkdir(exist_ok=True)
    print("Downloading the pinned FSDD release...", flush=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "spoken-digit-study"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if len(parts) == 3 and parts[1] == "recordings" and parts[2].endswith(".wav"):
                destination = DATA / "recordings" / parts[2]
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(archive.read(member))
            elif len(parts) == 2 and parts[1] in {"README.md", "LICENSE"}:
                (DATA / f"UPSTREAM_{parts[1]}").write_bytes(archive.read(member))
    (DATA / "manifest.json").write_text(json.dumps({
        "source": URL, "commit": COMMIT, "tag": "v1.0.10",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "license": "CC-BY-SA-4.0",
    }, indent=2) + "\n")


def read_audio(path):
    rate, values = wavfile.read(path)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Expected integer mono PCM: {path}")
    waveform = values.astype(np.float64) / max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max)
    if rate != 8000 or len(waveform) == 0:
        raise ValueError(f"Unexpected sample rate or empty recording: {path}")
    return rate, waveform


def mel_filters(rate=8000, nfft=256, count=26):
    """Triangular filters on a mel-spaced frequency axis."""
    mel_top = 2595 * np.log10(1 + (rate / 2) / 700)
    frequencies = 700 * (10 ** (np.linspace(0, mel_top, count + 2) / 2595) - 1)
    fft_frequencies = np.fft.rfftfreq(nfft, 1 / rate)
    filters = []
    for left, center, right in zip(frequencies[:-2], frequencies[1:-1], frequencies[2:]):
        filters.append(np.maximum(0, np.minimum(
            (fft_frequencies - left) / (center - left),
            (right - fft_frequencies) / (right - center))))
    return np.asarray(filters)


FILTERS = mel_filters()


def log_mel_frames(waveform, rate=8000):
    """Return the time-resolved representation before baseline pooling."""
    signal = waveform - waveform.mean()
    signal = np.r_[signal[0], signal[1:] - 0.97 * signal[:-1]]
    frame_length, hop = 200, 80
    count = max(1, 1 + int(np.ceil((len(signal) - frame_length) / hop)))
    signal = np.pad(signal, (0, max(0, (count - 1) * hop + frame_length - len(signal))))
    frames = signal[np.arange(count)[:, None] * hop + np.arange(frame_length)]
    power = np.abs(np.fft.rfft(frames * np.hamming(frame_length), n=256)) ** 2 / 256
    return np.log(np.maximum(power @ FILTERS.T, 1e-12))


def features(waveform, rate=8000):
    """25-ms frames, 10-ms hop; MFCC 1..13 and their temporal differences."""
    duration = len(waveform) / rate
    log_mel = log_mel_frames(waveform, rate)
    count = len(log_mel)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:14]
    # Drop coefficient 0, which is strongly affected by absolute recording level.
    delta = np.gradient(mfcc, axis=0) if count > 1 else np.zeros_like(mfcc)
    acceleration = np.gradient(delta, axis=0) if count > 1 else np.zeros_like(delta)
    vector = np.concatenate([stat for array in (mfcc, delta, acceleration)
                             for stat in (array.mean(axis=0), array.std(axis=0))] + [[duration]])
    if not np.isfinite(vector).all() or vector.shape != (79,):
        raise ValueError("Invalid features")
    return vector


def add_noise(waveform, snr_db, rng):
    """Add Gaussian noise with the specified measured signal/noise power ratio."""
    noise = rng.normal(size=len(waveform))
    signal_power = np.mean(waveform ** 2)
    noise *= np.sqrt(signal_power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return waveform + noise


def model():
    # These parameters are fixed before evaluation; no test-set tuning.
    return make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale"))


def score(labels, predicted):
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=np.arange(10), average="macro", zero_division=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    paths = sorted((DATA / "recordings").glob("*.wav"))
    if not paths:
        raise SystemExit("Run python study.py --download first.")
    OUT.mkdir(exist_ok=True)
    labels, groups, indices, clean, noisy = [], [], [], [], {10: [], 0: []}
    rng = np.random.default_rng(SEED)
    for index, path in enumerate(paths):
        digit, speaker, repeat = path.stem.split("_")
        rate, waveform = read_audio(path)
        labels.append(int(digit)); groups.append(speaker); indices.append(int(repeat))
        clean.append(features(waveform, rate))
        for snr in noisy:
            noisy[snr].append(features(add_noise(waveform, snr, rng), rate))
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{len(paths)} recordings", flush=True)
    X, y, groups, indices = np.asarray(clean), np.asarray(labels), np.asarray(groups), np.asarray(indices)
    noisy = {snr: np.asarray(values) for snr, values in noisy.items()}
    if len(paths) != 3000 or len(set(groups)) != 6:
        raise ValueError("Pinned dataset size differs from expected 3000 recordings / 6 speakers")

    known_test = indices < 5
    known_model = model().fit(X[~known_test], y[~known_test])
    known_score = score(y[known_test], known_model.predict(X[known_test]))
    predictions = {"clean": np.empty_like(y), "10dB": np.empty_like(y), "0dB": np.empty_like(y)}
    folds = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        fitted = model().fit(X[train], y[train])
        speaker = str(groups[test][0])
        row = {"held_out_speaker": speaker, "n_train": len(train), "n_test": len(test)}
        for condition, matrix in [("clean", X), ("10dB", noisy[10]), ("0dB", noisy[0])]:
            pred = fitted.predict(matrix[test])
            predictions[condition][test] = pred
            row[condition] = score(y[test], pred)
        folds.append(row)
        print(f"Held-out {speaker}: {row['clean']}", flush=True)
    results = {
        "dataset": json.loads((DATA / "manifest.json").read_text()),
        "n_recordings": len(paths), "speakers": sorted(set(groups)), "seed": SEED,
        "method": "79 features: MFCC 1-13 and differences, mean/std; duration. StandardScaler + RBF SVC C=10.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "recording_split_known_speakers": known_score,
        "leave_one_speaker_out": {condition: score(y, pred) for condition, pred in predictions.items()},
        "folds": folds,
        "noise_note": "Training uses clean audio only. Test noise is synthetic Gaussian noise; not café noise.",
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(OUT / "predictions.npz", labels=y, groups=groups, filenames=np.asarray([p.name for p in paths]), **predictions)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar([f["held_out_speaker"] for f in folds], [f["clean"]["accuracy"] for f in folds], color="#326d93")
    axes[0].set(ylabel="Accuracy", ylim=(0, 1), title="Each speaker excluded from training")
    axes[0].tick_params(axis="x", rotation=30)
    conditions = ["clean", "10dB", "0dB"]
    axes[1].plot(conditions, [results["leave_one_speaker_out"][c]["accuracy"] for c in conditions], "o-", color="#326d93")
    axes[1].set(ylabel="Accuracy", ylim=(0, 1), title="Clean-trained model; noisy test recordings")
    figure.savefig(OUT / "evaluation.png", dpi=160); plt.close(figure)
    matrix = confusion_matrix(y, predictions["clean"], labels=np.arange(10), normalize="true")
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted digit", ylabel="True digit", title="Unseen-speaker confusion matrix")
    figure.colorbar(image, ax=axis, label="Fraction of true-class recordings")
    figure.savefig(OUT / "confusion.png", dpi=160); plt.close(figure)
    example = next(path for path in paths if path.name == "7_jackson_0.wav")
    rate, waveform = read_audio(example)
    frequency, time, power = spectrogram(waveform, fs=rate, nperseg=200, noverlap=120, nfft=256)
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
    axes[0].plot(np.arange(len(waveform)) / rate, waveform, color="#326d93")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Example: spoken digit seven")
    image = axes[1].pcolormesh(time, frequency, 10 * np.log10(np.maximum(power, 1e-12)), shading="auto", cmap="magma")
    axes[1].set(xlabel="Time (s)", ylabel="Frequency (Hz)")
    figure.colorbar(image, ax=axes[1], label="Power (dB)")
    figure.savefig(OUT / "audio-example.png", dpi=160); plt.close(figure)
    print(json.dumps({"known_speakers": known_score, "unseen_speakers": results["leave_one_speaker_out"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
