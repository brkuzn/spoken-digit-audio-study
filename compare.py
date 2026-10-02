"""Time-resolved FSDD classifiers; independently implemented, with source credits."""
import json
import numpy as np
from scipy.fft import dct
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from study import DATA, OUT, SEED, read_audio, add_noise, log_mel_frames, score

FRAMES, MAX_EPOCHS, BATCH = 64, 30, 64
REFERENCE = 'https://es.mathworks.com/help/signal/ug/spoken-digit-recognition-with-custom-log-spectrogram-layer-and-deep-learning.html'

def resample_time(matrix, frames=FRAMES):
    """Interpolate time; retain the ordered frequency trajectory."""
    old, new = np.linspace(0,1,len(matrix)), np.linspace(0,1,frames)
    return np.stack([np.interp(new,old,col) for col in matrix.T]).astype(np.float32)

def representations(waveform):
    mel = log_mel_frames(waveform)
    mfcc = dct(mel,type=2,axis=1,norm='ortho')[:,1:14]
    temporal = resample_time(mfcc,32).ravel()
    image = resample_time(mel)
    image -= image.mean()  # One scalar per clip reduces recording-level dependence.
    return temporal, image[None]

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,12,3,padding=1),nn.BatchNorm2d(12),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(12,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),
            nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Dropout(.2),
            nn.Linear(576,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):
        return self.layers(x)

def fit_cnn(X,y,seed,epochs=MAX_EPOCHS,validation=None):
    torch.manual_seed(seed)
    net = SmallCNN()
    mean,std = float(X.mean()),max(float(X.std()),1e-6)
    data = TensorDataset(torch.from_numpy((X-mean)/std),torch.from_numpy(y).long())
    loader = DataLoader(data,batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optim = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    loss_fn = nn.CrossEntropyLoss()
    best_loss,best_epoch = float('inf'),1
    for epoch in range(1,epochs+1):
        net.train()
        for xb,yb in loader:
            optim.zero_grad();loss_fn(net(xb),yb).backward();optim.step()
        if validation is not None:
            vx,vy = validation
            net.eval()
            with torch.no_grad():
                loss = float(loss_fn(net(torch.from_numpy((vx-mean)/std)),torch.from_numpy(vy).long()))
            if loss<best_loss:
                best_loss,best_epoch = loss,epoch
    net.eval()
    return net,mean,std,best_epoch

def cnn_predict(fitted,X):
    net,mean,std,_ = fitted
    with torch.no_grad():
        batches = np.array_split(X,max(1,int(np.ceil(len(X)/BATCH))))
        return np.concatenate([net(torch.from_numpy((b-mean)/std)).argmax(1).numpy() for b in batches])

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths = sorted((DATA/'recordings').glob('*.wav'))
    if len(paths)!=3000:
        raise SystemExit('Run study.py --download first.')
    cache = OUT/'sequence-features.npz'
    cache_manifest = OUT/'sequence-features-manifest.json'
    metadata = {'version':1, 'seed':SEED,
                'dataset_commit':json.loads((DATA/'manifest.json').read_text())['commit'],
                'filenames':[p.name for p in paths]}
    cache_valid = (cache.exists() and cache_manifest.exists()
                   and json.loads(cache_manifest.read_text()) == metadata)
    if not cache_valid:
        arrays = {k:[] for k in ('clean_svc','clean_cnn','10dB_svc','10dB_cnn','0dB_svc','0dB_cnn')}
        rng = np.random.default_rng(SEED)
        for index,path in enumerate(paths):
            _,wave = read_audio(path)
            for condition,signal in [('clean',wave),('10dB',add_noise(wave,10,rng)),('0dB',add_noise(wave,0,rng))]:
                a,b = representations(signal)
                arrays[f'{condition}_svc'].append(a);arrays[f'{condition}_cnn'].append(b)
            if (index+1)%500==0:
                print(f'Sequence features {index+1}/3000',flush=True)
        np.savez_compressed(cache,**{k:np.asarray(v) for k,v in arrays.items()})
        cache_manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    parsed = [p.stem.split('_') for p in paths]
    y = np.array([int(p[0]) for p in parsed])
    groups = np.array([p[1] for p in parsed])
    repeats = np.array([int(p[2]) for p in parsed])
    matrices = np.load(cache)
    predictions = {k:np.full(len(y),-1,dtype=int) for k in matrices.files}
    folds = []
    for fold,(train,test) in enumerate(LeaveOneGroupOut().split(y,y,groups)):
        assert set(groups[train]).isdisjoint(groups[test])
        speaker = str(groups[test][0])
        svc = GridSearchCV(make_pipeline(StandardScaler(),SVC(gamma='scale')),
                           {'svc__C':[1,10,100]},cv=LeaveOneGroupOut(),scoring='accuracy',n_jobs=1)
        svc.fit(matrices['clean_svc'][train],y[train],groups=groups[train])
        # Predetermined inner voice selects epoch count. Outer voice is never used.
        voices = sorted(set(groups[train]))
        val_voice = voices[fold%len(voices)]
        val = train[groups[train]==val_voice];inner = train[groups[train]!=val_voice]
        assert set(groups[inner]).isdisjoint(groups[val])
        seed = SEED+fold
        selected = fit_cnn(matrices['clean_cnn'][inner],y[inner],seed,
                           validation=(matrices['clean_cnn'][val],y[val]))[3]
        cnn = fit_cnn(matrices['clean_cnn'][train],y[train],seed,epochs=selected)
        row = {'held_out_speaker':speaker,'n_train':len(train),'n_test':len(test),
               'svc_C':svc.best_params_['svc__C'],'svc_inner_accuracy':float(svc.best_score_),
               'cnn_validation_speaker':val_voice,'cnn_selected_epochs':selected,
               'inner_speakers':sorted(set(groups[inner])),'train_speakers':voices}
        for condition in ('clean','10dB','0dB'):
            for name,pred in [('svc',svc.predict(matrices[f'{condition}_svc'][test])),
                              ('cnn',cnn_predict(cnn,matrices[f'{condition}_cnn'][test]))]:
                key = f'{condition}_{name}';predictions[key][test] = pred;row[key] = score(y[test],pred)
        folds.append(row)
        print(json.dumps(row),flush=True)
        (OUT/'comparison-progress.json').write_text(json.dumps(folds,indent=2)+'\n')
    known_test = repeats<5
    known_svc = make_pipeline(StandardScaler(),SVC(C=10,gamma='scale')).fit(matrices['clean_svc'][~known_test],y[~known_test])
    known_cnn = fit_cnn(matrices['clean_cnn'][~known_test],y[~known_test],SEED+100,epochs=MAX_EPOCHS)
    result = {
        'baseline_source':'metrics.json','dataset_commit':'d6938f9bf1545aa66d8489fc9f1385a7abd64282',
        'n_recordings':3000,'seed':SEED,'torch_version':torch.__version__,
        'methods':{'svc':'13 MFCC coefficients, interpolated to32 frames; fold-local scaler and RBF SVC; C in [1,10,100] by inner leave-one-speaker-out CV.',
                   'cnn':'26 log-mel bands, interpolated to64 frames, clip scalar mean subtraction; 3-block 2D CNN; AdamW lr=.001 weight_decay=.001 dropout=.2 batch64. One inner speaker selects epochs (1..30), then refit on all5 training voices.'},
        'training':'Clean audio only; no noise augmentation; CNN scaling fitted on training clips only.',
        'development_note':'Exploratory follow-up after inspecting baseline scores. Outer test voices never select hyperparameters/checkpoints; this reused dataset is not a fresh final test.',
        'cnn_reference':REFERENCE,
        'reference_note':'Independent PyTorch implementation of the log-spectrogram CNN idea. Features and architecture differ; not an exact reproduction.',
        'recording_split_known_speakers':{'svc':score(y[known_test],known_svc.predict(matrices['clean_svc'][known_test])),
                                         'cnn':score(y[known_test],cnn_predict(known_cnn,matrices['clean_cnn'][known_test]))},
        'leave_one_speaker_out':{k:score(y,v) for k,v in predictions.items()},'folds':folds}
    assert all(np.all(p>=0) for p in predictions.values())
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'sequence-predictions.npz',labels=y,groups=groups,filenames=np.array([p.name for p in paths]),**predictions)
    baseline = json.loads((OUT/'metrics.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x = np.arange(6)
    for off,label,values,color in [(-.24,'Pooled MFCC + SVC',[f['clean']['accuracy'] for f in baseline['folds']],'#aaa'),
        (0,'Temporal MFCC + SVC',[f['clean_svc']['accuracy'] for f in folds],'#237a57'),
        (.24,'Log-mel CNN',[f['clean_cnn']['accuracy'] for f in folds],'#326d93')]:
        axes[0].bar(x+off,values,width=.24,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=[f['held_out_speaker'] for f in folds],ylim=(0,1),ylabel='Accuracy',title='New speaker: clean audio')
    axes[0].tick_params(axis='x',rotation=30)
    conditions = ['clean','10dB','0dB']
    axes[1].plot(conditions,[baseline['leave_one_speaker_out'][c]['accuracy'] for c in conditions],'o-',label='Pooled MFCC + SVC',color='#aaa')
    for name,label,color in [('svc','Temporal MFCC + SVC','#237a57'),('cnn','Log-mel CNN','#326d93')]:
        axes[1].plot(conditions,[result['leave_one_speaker_out'][f'{c}_{name}']['accuracy'] for c in conditions],'o-',label=label,color=color)
    axes[1].set(ylim=(0,1),ylabel='Accuracy',title='Clean training; synthetic noise at test')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'known':result['recording_split_known_speakers'],'unseen':result['leave_one_speaker_out']},indent=2),flush=True)

if __name__=='__main__':
    main()
