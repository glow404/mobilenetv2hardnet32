#!/usr/bin/env python3
"""V8/V9 共用实现：在原始 CSV 复数域完成 Base/Raw 相位处理。

为什么不经过 ``diff_final -> pic_final``：五个 Rgd 图若分别转成 uint8，
Base 与 Raw 的相对幅值会丢失；而相位对齐必须在原始十张 CSV 的复数域完成。

本文件只放两条待比较的新思路：

* V8（``independent``）：wo、wi 分别估计初始相位，分别五相位解调后相减；
* V9（``same-align``）：wo、wi 使用同一初始相位，复数解调后先全局相位对齐
  wi 到 wo，再相减。

常用入口不是本文件，而是 ``diffaractionv8.py`` 与
``diffaractionv9.py``。本文件保留为避免两份数学实现发生漂移。
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from diffaractionv7 import (DiffractionCalibrator, FREQ, DELTAT, SAMPNUM,
 TOTAL_THICKNESS, REQUIRED_RGDS,
 PHASE_MIN_MODULATION_PERCENTILE, PHASE_WEIGHT_POWER,
 DO_LOCAL_BLOCK_CORRECTION, LOCAL_BLOCK_ROW, LOCAL_BLOCK_COL)

PATTERN=re.compile(r"(wi|wo)-pair_(\d+).*?Rgd=(1421|1423|1425|1427|1429)",re.I)

def read_csv(path:Path)->np.ndarray:
    try: return pd.read_csv(path,header=None,encoding='utf-8-sig',low_memory=False).to_numpy(np.float32)
    except UnicodeDecodeError: return pd.read_csv(path,header=None,encoding='gbk',low_memory=False).to_numpy(np.float32)

def pairs(folder:Path):
    out={}
    for p in folder.glob('*.csv'):
        m=PATTERN.search(p.name)
        if m: out.setdefault(int(m.group(2)),{}).setdefault(m.group(1).lower(),{})[int(m.group(3))]=p
    return out

def robust_phase_alignment(base:np.ndarray, raw:np.ndarray)->tuple[np.ndarray,float,float]:
    # Magnitude excludes weak / unstable pixels from determining global phase.
    w=np.sqrt(np.abs(base)*np.abs(raw)); keep=w>np.percentile(w,35)
    corr=np.sum((base*np.conj(raw))[keep])
    if abs(corr)<1e-12: return raw,0.,0.
    angle=float(np.angle(corr)); coherence=float(abs(corr)/(np.abs(base[keep]*np.conj(raw[keep])).sum()+1e-12))
    return raw*np.exp(1j*angle),angle,coherence

def reconstruct(cal, wo, wi, scheme):
    if scheme=='independent':
        a=cal.estimate_initial_phase(wo,DELTAT,FREQ,PHASE_MIN_MODULATION_PERCENTILE,PHASE_WEIGHT_POWER)['phase']
        b=cal.estimate_initial_phase(wi,DELTAT,FREQ,PHASE_MIN_MODULATION_PERCENTILE,PHASE_WEIGHT_POWER)['phase']
        c=cal.load_and_demodulate(wo,DELTAT,FREQ,SAMPNUM,a)-cal.load_and_demodulate(wi,DELTAT,FREQ,SAMPNUM,b)
        meta={'wo_initial_phase':a,'wi_initial_phase':b,'alignment_phase':None,'alignment_coherence':None}
    elif scheme=='same-align':
        # The common phase is intentionally zero: any common rotation cancels
        # from relative alignment and final optimize_phase handles display sign.
        cb=cal.load_and_demodulate(wo,DELTAT,FREQ,SAMPNUM,0.)
        cr=cal.load_and_demodulate(wi,DELTAT,FREQ,SAMPNUM,0.)
        cr,angle,coh=robust_phase_alignment(cb,cr); c=cb-cr
        meta={'wo_initial_phase':0.,'wi_initial_phase':0.,'alignment_phase':angle,'alignment_coherence':coh}
    else:
        raise ValueError(f'未知方案: {scheme}')
    c=cal.global_rowcol_correction(c)
    if DO_LOCAL_BLOCK_CORRECTION:
        c=cal.local_block_mean_correction(
            c, block_height=LOCAL_BLOCK_ROW, block_width=LOCAL_BLOCK_COL)
    return cal.deconvolve_diffraction_optimized(c,TOTAL_THICKNESS),meta

def write(path:Path,image:np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True); ok,b=cv2.imencode(path.suffix,image)
    if not ok: raise RuntimeError(path)
    b.tofile(str(path))

def main(default_scheme='both'):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
 p.add_argument('--scheme',choices=('independent','same-align','both'),default=default_scheme)
 p.add_argument('--flat-output',action='store_true',help='单方案时直接输出为 <output>/<identity>/pair_N.bmp。')
 p.add_argument('--folder',action='append',default=[]);p.add_argument('--limit-pairs',type=int,default=0)
 a=p.parse_args();
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 schemes=('independent','same-align') if a.scheme=='both' else (a.scheme,)
 a.output_root.mkdir(parents=True); rows=[]
 folders=[x for x in sorted(a.input_root.iterdir()) if x.is_dir() and (not a.folder or x.name in a.folder)]
 for folder in folders:
  grouped=pairs(folder)
  for pair,items in sorted(grouped.items()):
   if a.limit_pairs and pair>a.limit_pairs: continue
   if set(items)!= {'wo','wi'} or any(r not in items['wo'] or r not in items['wi'] for r in REQUIRED_RGDS): continue
   wo=[read_csv(items['wo'][r]) for r in REQUIRED_RGDS];wi=[read_csv(items['wi'][r]) for r in REQUIRED_RGDS]
   if any(x.shape!=wo[0].shape for x in wo+wi): continue
   for scheme in schemes:
    cal=DiffractionCalibrator(); result,meta=reconstruct(cal,wo,wi,scheme)
    # optimize_phase contains the established V7 deconvolution display phase.
    path=(a.output_root/folder.name/f'pair_{pair}.bmp' if a.flat_output else a.output_root/scheme/folder.name/f'pair_{pair}.bmp')
    # V7's legacy optimize_phase writes directly with cv2.imwrite and does not
    # create parents itself.
    path.parent.mkdir(parents=True, exist_ok=True)
    final=cal.optimize_phase(result,str(path))
    rows.append({'folder':folder.name,'pair':pair,'scheme':scheme,**meta,
                 'output_std':float(np.std(final)),'output_laplacian':float(cv2.Laplacian(final.astype(np.float32),cv2.CV_32F).var())})
    print(f'{scheme} {folder.name}/pair_{pair}',flush=True)
 (a.output_root/'phase_metadata.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(f'completed={len(rows)} output={a.output_root}')
if __name__=='__main__':main()
