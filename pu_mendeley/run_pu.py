"""PU cross-domain experiment adapted from TIMC-26-1736 supplementary code.

Uses real third-party Paderborn FFT arrays from Deng (2023), DOI
10.17632/65d3pzth7v.1. Their equivalence to the authors' preprocessing is not
verified. Fixed 15 epochs, with target-domain evaluation only once at the end.
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=AdaptiveChebyshevKANLayer(1500,64,5)
        self.layer2=AdaptiveChebyshevKANLayer(64,8,5)
    def forward(self,x):
        return self.layer2(self.layer1(x.squeeze(1)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent.parent/'data'/'PU'/'bearingset'/'raw')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    args=p.parse_args()
    if args.epochs<1 or args.threads<1: p.error('epochs and threads must be positive')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available(): p.error('MPS unavailable')
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device=='cuda': torch.cuda.manual_seed_all(args.seed)
    paths=[args.data_dir/f for f in ['FFT_0.npy','FFT_1.npy','Label.npy']]
    if any(not f.is_file() for f in paths): raise FileNotFoundError(f'Required files FFT_0.npy, FFT_1.npy, Label.npy in {args.data_dir}')
    x0,x1,labels=(np.load(f,allow_pickle=False) for f in paths)
    if x0.shape!=(8000,1,1500) or x1.shape!=(8000,1,1500) or labels.shape!=(8000,8):
        raise ValueError('Expected arrays (8000,1,1500), (8000,1,1500), (8000,8)')
    if not all(np.isfinite(v).all() for v in (x0,x1,labels)):
        raise ValueError('Array contains non-finite data')
    if not np.all((labels==0)|(labels==1)) or not np.all(labels.sum(axis=1)==1):
        raise ValueError('Labels must be one-hot encoded')
    classes=np.argmax(labels,axis=1)
    counts=np.bincount(classes,minlength=8)
    print('Class counts:', counts.tolist(),flush=True)
    src=torch.FloatTensor(x0); tgt=torch.FloatTensor(x1)
    del x0,x1,labels
    # Match supplied script: source scalar statistics across sample and signal dimensions.
    mean=src.mean(dim=(0,2),keepdim=True); std=src.std(dim=(0,2),keepdim=True)+1e-8
    src=(src-mean)/std; tgt=(tgt-mean)/std
    y=torch.LongTensor(classes)
    train=DataLoader(TensorDataset(src,y),batch_size=64,shuffle=True)
    test=DataLoader(TensorDataset(tgt,y),batch_size=64,shuffle=False)
    model=Classifier().to(args.device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]; start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); correct=0; loss_total=0
        epoch_start=time.perf_counter()
        for xb,yb in train:
            xb=xb.to(args.device); yb=yb.to(args.device)
            optimizer.zero_grad(); out=model(xb); loss=criterion(out,yb)
            loss.backward(); optimizer.step()
            correct+=(out.argmax(1)==yb).sum().item(); loss_total+=loss.item()*len(yb)
        item={'epoch':epoch+1,'train_loss':loss_total/len(y),'train_accuracy':correct/len(y),
              'seconds':time.perf_counter()-epoch_start}
        history.append(item)
        print(f"Epoch {epoch+1}/{args.epochs} loss={item['train_loss']:.4f} train_acc={item['train_accuracy']:.4f} sec={item['seconds']:.1f}",flush=True)
    model.eval(); correct=0; n=0
    with torch.no_grad():
        for xb,yb in test:
            out=model(xb.to(args.device)); correct+=(out.argmax(1).cpu()==yb).sum().item(); n+=len(yb)
    accuracy=correct/n
    print(f'Target FFT_1 accuracy={accuracy:.6f} ({correct}/{n})',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result={'source':'Deng (2023), DOI 10.17632/65d3pzth7v.1',
            'caveat':'Third-party preprocessing; equivalence to manuscript authors\' arrays unverified',
            'seed':args.seed,'epochs':args.epochs,'device':args.device,'threads':args.threads,
            'class_counts':counts.tolist(),'source_samples':len(y),'target_samples':n,
            'target_accuracy':accuracy,'target_correct':correct,
            'elapsed_seconds':time.perf_counter()-start,'training_history':history,
            'test_protocol':'FFT_0 training, FFT_1 evaluation once after fixed final epoch'}
    (args.output_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'source_mean':mean,'source_std':std,'seed':args.seed},args.output_dir/'model.pt')

if __name__=='__main__': main()
