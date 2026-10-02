# What the higher published scores measure

Reviewed on 2 October 2026. The numbers below are reported by the authors; their
experiments were not reproduced locally. Our runs use one pinned 3,000-clip FSDD
release, so older datasets and different splits are not directly comparable.

| Source | Reported accuracy | Representation and model | Evaluation |
|---|---:|---|---|
| [MathWorks example](https://es.mathworks.com/help/signal/ug/spoken-digit-recognition-with-custom-log-spectrogram-layer-and-deep-learning.html) | 99% | Log spectrogram and a 2D CNN | A shuffled, class-balanced 80/20 recording split on an older 2,000-clip version. Speakers are not explicitly held out. |
| [Mathis Lacombe's project](https://github.com/Mathos34/spoken-digits-asr) | 98.0% | Log-mel features, Conv1D + bidirectional LSTM, CTC | The official 2,700/300 recording-index split. The training script selects the checkpoint with the lowest **test** CER. |
| [Aniruddha Pal's project](https://github.com/aniruddhapal211316/spoken_digit_recognition) | 97.33% | Time-resolved MFCCs and a bidirectional LSTM | Random 80/10/10 recording split; `dataset.py` uses `random_split`, without speaker grouping. |

For our initial pooled-MFCC classifier, accuracy on the official recording-index
split was 96.33%. Its 70.50% result came from excluding an entire speaker from
training. Comparing that figure directly with these 97–99% figures would mix two
different questions.

The useful methodological idea is to keep the changing frequency structure of
the word. Pooling an entire clip into means and standard deviations loses much
of its internal order. Both recurrent networks and spectrogram CNNs can use that
structure.

`compare.py` tests two follow-ups: time-interpolated MFCCs with an SVC, and a small
2D CNN on log-mel images. The CNN uses the spectrogram-CNN idea illustrated by
MathWorks; its representation, architecture and training settings are independently
implemented and differ from that example. No external source code was copied.
The temporal SVC is a simpler comparator for retaining within-word order.

SVC regularisation is selected with inner leave-one-speaker-out validation on the
five training voices. For the CNN, one predetermined training voice selects the
epoch count before the model is refitted on all five voices. The outer test voice
is not used for either choice. All models train on clean clips; the same seeded
noisy waveforms are used at evaluation.

This remains an exploratory study: the dataset has already been inspected and
used during development. A larger, untouched speaker set would be needed for a
fresh final generalisation estimate. The comparison does not establish which
single design change caused any gain.

## Code inspected

- Mathos34/spoken-digits-asr, commit
  [`16d19a3ea5fdcdafad4db7ee84b825e0d51f73fa`](https://github.com/Mathos34/spoken-digits-asr/tree/16d19a3ea5fdcdafad4db7ee84b825e0d51f73fa):
  `src/model.py` and `train.py`. Test-based checkpoint selection is visible in the
  training loop; it was not adopted here.
- aniruddhapal211316/spoken_digit_recognition, commit
  [`147f43538a075a728690ece8fe6e1e268188b24b`](https://github.com/aniruddhapal211316/spoken_digit_recognition/tree/147f43538a075a728690ece8fe6e1e268188b24b):
  `dataset.py`, `model.py`, `train.py`.
- The MathWorks page supplies the CNN layers, training settings, random split and
  reported accuracy directly. Its prose describes six speakers, while the
  displayed class counts sum to 2,000 clips; this review uses its explicit clip
  count and does not assume that its dataset is the same as ours.

## Follow-up after the first comparison

The unnormalised temporal SVC reached 78.27% unseen-speaker clean accuracy; the
CNN reached 74.10%. Both exceeded the 70.50% baseline overall but were less
robust to Gaussian noise. Neither improved George's clean result.

Pal's `Standardize` transform normalises a `(coefficients, time)` matrix along
`axis=0`: across coefficients within each frame. It does not apply time-axis
CMVN. An independently implemented variant of that step was added to the
32-frame MFCC/SVC representation in `normalized.py`. Unlike the source, this uses
13 coefficients and applies normalisation after time interpolation.

That follow-up reached 82.33% on clean unseen-speaker clips, with inner validation
for SVC regularisation. Its 10 dB and 0 dB results were 31.30% and 14.67%, below
the baseline's 47.30% and 22.10%. It is retained as the stronger clean-audio model,
with the noise regression visible. It was added after the earlier outer results
were inspected, so the dataset serves as exploratory development evidence.

A preliminary fixed-C check also tried time-axis cepstral mean normalisation
and mean/variance normalisation on the time-interpolated features (73.07% and
73.50% clean accuracy). Those probes were not selected for the final follow-up,
are not independent evaluations, and have no retained per-clip predictions.
See [Kaldi's transform documentation](https://kaldi-asr.org/doc/transform.html)
for the conventional CMVN definition.
