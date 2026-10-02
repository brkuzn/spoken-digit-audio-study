"""Checks for noise calibration and evaluation integrity."""
import json
import numpy as np
from study import ROOT, add_noise, features

wave = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
for target in (0, 10):
    noisy = add_noise(wave, target, np.random.default_rng(8))
    measured = 10 * np.log10(np.mean(wave ** 2) / np.mean((noisy - wave) ** 2))
    assert abs(measured - target) < 1e-10
assert np.isfinite(features(np.zeros(80))).all()
results = json.loads((ROOT / "results/metrics.json").read_text())
predictions = np.load(ROOT / "results/predictions.npz")
assert len(predictions["labels"]) == results["n_recordings"] == 3000
assert len(set(predictions["filenames"])) == 3000
assert len(results["folds"]) == 6
for fold in results["folds"]:
    assert fold["n_train"] == 2500 and fold["n_test"] == 500
assert sum(fold["n_test"] for fold in results["folds"]) == 3000
print("Noise calibration, complete predictions, and held-out fold sizes: PASS")

# Validate the follow-up's normalisation and every saved held-out prediction.
from normalized import normalise_frames
from study import score
from compare import representations

rng = np.random.default_rng(12)
test_matrix = rng.normal(size=(3, 416)).astype(np.float32)
normalised = normalise_frames(test_matrix).reshape(3, 13, 32)
assert np.allclose(normalised.mean(axis=1), 0, atol=1e-6)
assert np.allclose(normalised.std(axis=1), 1, atol=1e-6)
assert np.isfinite(normalise_frames(np.zeros((2, 416), dtype=np.float32))).all()
assert np.allclose(normalise_frames(test_matrix[:1]), normalise_frames(test_matrix)[:1])
for wave in (np.zeros(80), np.sin(2*np.pi*440*np.arange(8000)/8000)):
    temporal, image = representations(wave)
    assert temporal.shape == (416,) and image.shape == (1, 26, 64)
    assert np.isfinite(temporal).all() and np.isfinite(image).all()
for report_name, prediction_name, mapping in [
    ('comparison.json', 'sequence-predictions.npz',
     {c+'_'+m: c+'_'+m for c in ('clean','10dB','0dB') for m in ('svc','cnn')}),
    ('normalized.json', 'normalized-predictions.npz', {c:c for c in ('clean','10dB','0dB')})]:
    report = json.loads((ROOT/'results'/report_name).read_text())
    saved = np.load(ROOT/'results'/prediction_name)
    assert np.array_equal(saved['filenames'], predictions['filenames'])
    assert np.array_equal(saved['labels'], predictions['labels'])
    assert np.array_equal(saved['groups'], predictions['groups'])
    assert len(report['folds']) == 6
    for key in mapping:
        assert np.isin(saved[key], np.arange(10)).all()
        assert score(saved['labels'], saved[key]) == report['leave_one_speaker_out'][key]
    for fold in report['folds']:
        speaker = fold['held_out_speaker']
        assert speaker not in fold['train_speakers']
        assert set(fold['train_speakers']) == set(saved['groups'])-{speaker}
        selected = saved['groups'] == speaker
        assert selected.sum() == fold['n_test'] == 500 and fold['n_train'] == 2500
        for key in mapping:
            assert score(saved['labels'][selected], saved[key][selected]) == fold[key]
        if 'cnn_validation_speaker' in fold:
            assert fold['cnn_validation_speaker'] in fold['train_speakers']
            assert fold['cnn_validation_speaker'] not in fold['inner_speakers']
            assert speaker not in fold['inner_speakers']
            assert 1 <= fold['cnn_selected_epochs'] <= 30
print('Follow-up normalisation, speaker separation, prediction coverage, and recomputed scores: PASS')
