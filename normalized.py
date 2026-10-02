"""Time-resolved MFCCs with per-frame coefficient standardisation.

Method inspiration: aniruddhapal211316/spoken_digit_recognition/dataset.py.
Independent implementation; fewer coefficients and time interpolation are used.
"""
import json
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from study import DATA, OUT, SEED, score

REFERENCE = 'https://github.com/aniruddhapal211316/spoken_digit_recognition/blob/147f43538a075a728690ece8fe6e1e268188b24b/dataset.py'

def normalise_frames(X):
    """Normalise coefficients within each time frame, independently per clip.

    This is across coefficients, not conventional time-axis CMVN. No statistics
    are shared across clips. Time interpolation precedes this normalisation.
    """
    frames = X.reshape(-1,13,32)
    mean = frames.mean(axis=1,keepdims=True)
    std = np.maximum(frames.std(axis=1,keepdims=True),1e-6)
    return ((frames-mean)/std).reshape(-1,416)

def main():
    paths = sorted((DATA/'recordings').glob('*.wav'))
    metadata = json.loads((OUT/'sequence-features-manifest.json').read_text())
    assert metadata['filenames'] == [p.name for p in paths]
    assert metadata['seed'] == SEED and metadata['version'] == 1
    assert metadata['dataset_commit'] == json.loads((DATA/'manifest.json').read_text())['commit']
    cache = np.load(OUT/'sequence-features.npz')
    assert len(paths)==3000
    X = {c:normalise_frames(cache[f'{c}_svc']) for c in ('clean','10dB','0dB')}
    y = np.array([int(p.name[0]) for p in paths])
    groups = np.array([p.stem.split('_')[1] for p in paths])
    repeats = np.array([int(p.stem.split('_')[2]) for p in paths])
    predictions = {c:np.full(len(y),-1,dtype=int) for c in X}
    folds = []
    for train,test in LeaveOneGroupOut().split(X['clean'],y,groups):
        assert set(groups[train]).isdisjoint(groups[test])
        m = GridSearchCV(make_pipeline(StandardScaler(),SVC(gamma='scale')),
                         {'svc__C':[1,10,100]},cv=LeaveOneGroupOut(),scoring='accuracy',n_jobs=1)
        m.fit(X['clean'][train],y[train],groups=groups[train])
        row = {'held_out_speaker':str(groups[test][0]),'train_speakers':sorted(set(groups[train])),
               'n_train':len(train),'n_test':len(test),'C':m.best_params_['svc__C'],
               'inner_accuracy':float(m.best_score_)}
        for c in X:
            pred = m.predict(X[c][test]);predictions[c][test] = pred;row[c] = score(y[test],pred)
        folds.append(row);print(json.dumps(row),flush=True)
    known_test = repeats<5
    # Select C on training recordings only, also grouping their speakers.
    known = GridSearchCV(make_pipeline(StandardScaler(),SVC(gamma='scale')),
                         {'svc__C':[1,10,100]},cv=LeaveOneGroupOut(),scoring='accuracy',n_jobs=1)
    known.fit(X['clean'][~known_test],y[~known_test],groups=groups[~known_test])
    results = {'n_recordings':3000,'dataset_commit':'d6938f9bf1545aa66d8489fc9f1385a7abd64282',
               'method':'13 MFCCs at32 ordered times, coefficient mean/std normalisation within each frame; fold-local StandardScaler + RBF SVC. C in[1,10,100], selected by inner leave-one-speaker-out validation.',
               'source':REFERENCE,'source_note':'Independent implementation inspired by framewise coefficient standardisation; source uses39 coefficients and BiLSTM. Here13 coefficients, interpolation and an SVC are used.',
               'training':'Clean clips only; no noise augmentation.',
               'development_note':'Exploratory follow-up. Frame normalisation was tried after inspecting other models on this same dataset; outer results are reused descriptive development evidence, not an untouched final evaluation. C selection does not use outer test speakers.',
               'recording_split_known_speakers':score(y[known_test],known.predict(X['clean'][known_test])),
               'known_split_C':known.best_params_['svc__C'],
               'leave_one_speaker_out':{c:score(y,p) for c,p in predictions.items()},'folds':folds}
    assert all(np.all(p>=0) for p in predictions.values())
    (OUT/'normalized.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(OUT/'normalized-predictions.npz',labels=y,groups=groups,
                        filenames=np.array([p.name for p in paths]),**predictions)
    baseline = json.loads((OUT/'metrics.json').read_text())
    compared = json.loads((OUT/'comparison.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x = np.arange(6)
    axes[0].bar(x-.18,[f['clean']['accuracy'] for f in baseline['folds']],width=.36,label='Pooled MFCC + SVC',color='#aaa')
    axes[0].bar(x+.18,[f['clean']['accuracy'] for f in folds],width=.36,label='Normalised temporal MFCC + SVC',color='#237a57')
    axes[0].set(xticks=x,xticklabels=[f['held_out_speaker'] for f in folds],ylim=(0,1),ylabel='Accuracy',title='Same held-out speakers: clean audio')
    axes[0].tick_params(axis='x',rotation=30)
    conditions = ['clean','10dB','0dB']
    for label,values,color in [('Pooled MFCC + SVC',[baseline['leave_one_speaker_out'][c]['accuracy'] for c in conditions],'#aaa'),
        ('Temporal MFCC + SVC',[compared['leave_one_speaker_out'][f'{c}_svc']['accuracy'] for c in conditions],'#ad8237'),
        ('Log-mel CNN',[compared['leave_one_speaker_out'][f'{c}_cnn']['accuracy'] for c in conditions],'#326d93'),
        ('Normalised temporal MFCC + SVC',[results['leave_one_speaker_out'][c]['accuracy'] for c in conditions],'#237a57')]:
        axes[1].plot(conditions,values,'o-',label=label,color=color)
    axes[1].set(ylim=(0,1),ylabel='Accuracy',title='Clean training; synthetic noisy test')
    axes[1].legend(fontsize=7)
    fig.savefig(OUT/'final-comparison.png',dpi=160);plt.close(fig)
    fig,axis = plt.subplots(figsize=(6,5),constrained_layout=True)
    im=axis.imshow(confusion_matrix(y,predictions['clean'],labels=np.arange(10),normalize='true'),vmin=0,vmax=1,cmap='Blues')
    axis.set(xticks=np.arange(10),yticks=np.arange(10),xlabel='Predicted digit',ylabel='True digit',title='Normalised temporal MFCCs: held-out speakers')
    fig.colorbar(im,ax=axis,label='Fraction of true-class recordings')
    fig.savefig(OUT/'normalized-confusion.png',dpi=160);plt.close(fig)
    print(json.dumps({'known':results['recording_split_known_speakers'],'unseen':results['leave_one_speaker_out']},indent=2),flush=True)

if __name__=='__main__':
    main()
