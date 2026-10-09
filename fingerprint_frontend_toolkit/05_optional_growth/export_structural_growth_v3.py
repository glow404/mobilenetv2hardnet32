#!/usr/bin/env python3
"""Run the V3 dual-stream growth model on saved V7 severe masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from train_ridge_gap_growth import tensor_fields
from train_structural_growth_v3 import DualStreamGrowthNet


def read(path):
    raw=np.fromfile(str(path),np.uint8); image=cv2.imdecode(raw,cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(path)
    return image


def write(path,image):
    path.parent.mkdir(parents=True,exist_ok=True);ok,buf=cv2.imencode(path.suffix or '.bmp',image)
    if not ok: raise RuntimeError(path)
    buf.tofile(str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--v7-root',type=Path,required=True);p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--limit-per-identity',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path.cwd()
    for n in ('v7_root','candidate_root','checkpoint','output_root'):
        if not getattr(a,n).is_absolute():setattr(a,n,root/getattr(a,n))
    if a.output_root.exists():raise FileExistsError(a.output_root)
    device=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=DualStreamGrowthNet().to(device).eval();model.load_state_dict(ckpt['model'])
    count=0
    for identity_dir in sorted((a.candidate_root/'images').iterdir()):
        if not identity_dir.is_dir():continue
        pairs=sorted(identity_dir.iterdir(),key=lambda x:int(x.name.rsplit('_',1)[-1]))
        if a.limit_per_identity:pairs=pairs[:a.limit_per_identity]
        for pair in pairs:
            filename=pair.name+'.bmp';source=read(a.v7_root/identity_dir.name/filename);mask=(read(pair/'mask.png')>0).astype(np.float32)
            theta,_,confidence=tensor_fields(source.astype(np.float32)/255.,mask);a2=2*theta
            texture=np.stack([source.astype(np.float32)/255.,mask])[None];structure=np.stack([np.sin(a2),np.cos(a2),confidence,mask])[None]
            with torch.inference_mode():prediction=model(torch.from_numpy(texture).to(device),torch.from_numpy(structure.astype(np.float32)).to(device))[0,0].cpu().numpy()
            output=np.where(mask>0,np.clip(prediction*255+.5,0,255).astype(np.uint8),source)
            write(a.output_root/identity_dir.name/filename,output);count+=1
    print(f'exported={count} root={a.output_root}')
if __name__=='__main__':main()
