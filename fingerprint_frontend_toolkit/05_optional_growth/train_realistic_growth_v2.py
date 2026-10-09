#!/usr/bin/env python3
"""V2 ridge-growth training with real severe-mask shapes and V7-like degradation.

Unlike V1's random-noise holes, V2 destroys ridge contrast with blur, bias,
speckle and bright/dark dropout inside masks harvested from real V7 failures.
The loss also matches oriented ridge response energy, preventing a low-contrast
conditional-mean completion from winning only by L1 error.
"""
from __future__ import annotations

import argparse, csv, json, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train_ridge_gap_growth import FlowGrowthUNet, tensor_fields, make_growth_mask, gabor_filters, grad_xy, dilate, save_preview


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise RuntimeError(f"Cannot read {path}")
    return image


def realistic_degrade(target: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Destroy local ridge evidence in the way observed in V7, not as white noise."""
    h, w = target.shape
    sigma = float(rng.uniform(0.9, 2.6))
    blurred = cv2.GaussianBlur(target, (0, 0), sigma)
    # Directional smear mimics interference/defocus more closely than isotropic blur.
    length = int(rng.integers(3, 11)); angle = float(rng.uniform(0, np.pi))
    kernel = np.zeros((length, length), np.float32)
    c = (length - 1) / 2; dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(kernel, (int(round(c-dx)), int(round(c-dy))), (int(round(c+dx)), int(round(c+dy))), 1.0, 1)
    kernel /= max(float(kernel.sum()), 1.0)
    blurred = cv2.filter2D(blurred, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    mean = float(target.mean())
    contrast = float(rng.uniform(0.10, 0.48))
    degraded = mean + contrast * (blurred - mean)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    field = (np.sin(xx * rng.uniform(.025, .10) + rng.uniform(0, 6.28)) +
             np.sin(yy * rng.uniform(.025, .10) + rng.uniform(0, 6.28)))
    degraded += field * rng.uniform(.025, .11)
    degraded += rng.normal(0, rng.uniform(.01, .06), target.shape)
    # Scattered saturation/black points and broad optical spots are both present in V7.
    for _ in range(int(rng.integers(2, 8))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        radius = int(rng.integers(1, 7))
        value = mean + rng.choice([-1.0, 1.0]) * rng.uniform(.10, .36)
        blob = np.zeros((h, w), np.float32); cv2.circle(blob, (x, y), radius, 1, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), max(radius / 2, .6))
        degraded = degraded * (1 - blob) + value * blob
    return (target * (1 - mask) + np.clip(degraded, 0, 1) * mask).astype(np.float32)


class RealisticGrowthDataset(Dataset):
    def __init__(self, files, real_masks, train, seed):
        self.files, self.real_masks, self.train, self.seed = files, real_masks, train, seed

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        target = read_gray(self.files[index]).astype(np.float32) / 255.0
        rng = np.random.default_rng(self.seed + index + (random.randrange(1 << 20) if self.train else 0))
        if self.train and rng.random() < .5: target = np.fliplr(target).copy()
        if self.train and rng.random() < .5: target = np.flipud(target).copy()
        if self.real_masks and rng.random() < .72:
            raw = read_gray(self.real_masks[int(rng.integers(len(self.real_masks)))])
            mask = cv2.resize((raw > 0).astype(np.uint8), target.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)
            if self.train and rng.random() < .5: mask = np.fliplr(mask).copy()
            if self.train and rng.random() < .5: mask = np.flipud(mask).copy()
            # keep a safety-compatible severity range; V1-shaped holes retain diversity.
            if not .01 < mask.mean() < .30: mask = make_growth_mask(target, rng)
        else:
            mask = make_growth_mask(target, rng)
        corrupted = realistic_degrade(target, mask, rng)
        tangent, _, confidence = tensor_fields(corrupted, mask)
        a2 = 2 * tangent
        inp = np.stack([corrupted, mask, np.sin(a2), np.cos(a2), confidence]).astype(np.float32)
        return torch.from_numpy(inp), torch.from_numpy(target[None].copy()), torch.from_numpy(mask[None])


def weighted_energy(x, mask):
    return torch.sqrt((x.square() * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6) + 1e-7)


def losses_v2(pred, target, mask, lambda_energy):
    support = dilate(mask)
    l1 = ((pred-target).abs()*mask).sum()/(mask.sum()+1e-6)
    px, py = grad_xy(pred); tx, ty = grad_xy(target)
    grad = (((px-tx).abs()+(py-ty).abs())*support).sum()/(2*support.sum()+1e-6)
    pn, tn = torch.sqrt(px.square()+py.square()+1e-6), torch.sqrt(tx.square()+ty.square()+1e-6)
    orientation = ((1-((px*tx+py*ty)/(pn*tn)).clamp(-1,1).abs())*support).sum()/(support.sum()+1e-6)
    bank = gabor_filters(pred.device, torch.float32)
    pg, tg = F.conv2d(pred, bank, padding=7), F.conv2d(target, bank, padding=7)
    gabor = ((pg-tg).abs()*support).sum()/(support.sum()*bank.shape[0]+1e-6)
    # Match ridge-band amplitude explicitly; L1/Gabor difference alone permits weak gray ridges.
    energy = (weighted_energy(pg, mask)-weighted_energy(tg, mask)).abs().mean()
    total = l1 + .25*grad + .20*orientation + 2.0*gabor + lambda_energy*energy
    return total, l1, grad, orientation, gabor, energy


@torch.no_grad()
def evaluate(model, loader, device, lambda_energy):
    model.eval(); totals=np.zeros(9, np.float64); previews=[]
    for inp,target,mask in loader:
        inp,target,mask=inp.to(device),target.to(device),mask.to(device)
        out=model(inp); result=out*mask+inp[:,:1]*(1-mask)
        parts=losses_v2(result.float(),target.float(),mask.float(),lambda_energy)
        mse=(((result-target).square()*mask).sum()/(mask.sum()+1e-6)).item()
        lap=target.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])[None,None]
        lr,lt=F.conv2d(result,lap,padding=1),F.conv2d(target,lap,padding=1)
        mr=(lr*mask).sum()/(mask.sum()+1e-6); mt=(lt*mask).sum()/(mask.sum()+1e-6)
        lvr=(((lr-mr).square()*mask).sum()/(mask.sum()+1e-6)).item(); lvt=(((lt-mt).square()*mask).sum()/(mask.sum()+1e-6)).item()
        totals += np.array([p.item() for p in parts]+[mse,lvr,lvt]) * target.shape[0]
        if len(previews)<8:
            for i in range(min(len(target),8-len(previews))): previews.append([target[i,0].cpu().numpy(),inp[i,0].cpu().numpy(),result[i,0].cpu().numpy(),mask[i,0].cpu().numpy()])
    v=totals/max(len(loader.dataset),1)
    return {"val_score":float(v[0]),"masked_l1":float(v[1]),"gradient_l1":float(v[2]),"orientation_loss":float(v[3]),"gabor_l1":float(v[4]),"ridge_energy_loss":float(v[5]),"masked_psnr_db":float(-10*np.log10(max(v[6],1e-12))),"completion_laplacian_variance":float(v[7]),"target_laplacian_variance":float(v[8])},previews


def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean-root',type=Path,default=Path('指纹识别数据集/不贴屏不贴膜/butieping'));p.add_argument('--mask-root',type=Path,default=Path('pipeline_outputs/growth_then_gan_workset_30_20260929/images'));p.add_argument('--init',type=Path,default=Path('fingergrowthV5/experiments/ridge_gap_growth_20260928_210211/best.pt'));p.add_argument('--output',type=Path,default=None);p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=8);p.add_argument('--max-train',type=int,default=0);p.add_argument('--max-val',type=int,default=0);p.add_argument('--lambda-energy',type=float,default=.8);p.add_argument('--lr',type=float,default=8e-5);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260929);a=p.parse_args()
    root=Path.cwd()
    for n in ('clean_root','mask_root','init'):
        if not getattr(a,n).is_absolute(): setattr(a,n,root/getattr(a,n))
    if a.output is None:a.output=root/'fingergrowthV5'/'experiments'/f'realistic_growth_v2_{time.strftime("%Y%m%d_%H%M%S")}'
    elif not a.output.is_absolute():a.output=root/a.output
    if a.output.exists():raise FileExistsError(a.output)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    files=sorted(a.clean_root.rglob('*.bmp'));groups={}
    for f in files:groups.setdefault(f.parents[1].name,[]).append(f)
    ids=sorted(groups); val_ids=set(ids[-max(2,round(.2*len(ids))):]);train=[f for k in ids if k not in val_ids for f in groups[k]];val=[f for k in ids if k in val_ids for f in groups[k]]
    random.Random(a.seed).shuffle(train);random.Random(a.seed+1).shuffle(val)
    if a.max_train:train=train[:a.max_train]
    if a.max_val:val=val[:a.max_val]
    masks=sorted(a.mask_root.rglob('mask.png'))
    if not masks:raise FileNotFoundError(f'No real masks under {a.mask_root}')
    dev=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu');a.output.mkdir(parents=True)
    (a.output/'split.json').write_text(json.dumps({'train_count':len(train),'val_count':len(val),'val_ids':sorted(val_ids),'mask_count':len(masks),'config':vars(a)},indent=2,default=str),encoding='utf-8')
    dl=lambda ds,sh:DataLoader(ds,batch_size=a.batch_size,shuffle=sh,num_workers=a.workers,pin_memory=dev.type=='cuda',persistent_workers=a.workers>0)
    train_loader,val_loader=dl(RealisticGrowthDataset(train,masks,True,a.seed),True),dl(RealisticGrowthDataset(val,masks,False,a.seed+100000),False)
    model=FlowGrowthUNet().to(dev)
    if a.init.exists(): model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model']);print(f'warm-start={a.init}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda');best=float('inf')
    fields=['epoch','train_loss','val_score','masked_l1','gradient_l1','orientation_loss','gabor_l1','ridge_energy_loss','masked_psnr_db','completion_laplacian_variance','target_laplacian_variance','lr']
    with (a.output/'train.csv').open('w',newline='',encoding='utf-8') as fp:
      writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
      for epoch in range(1,a.epochs+1):
        model.train();hist=[]
        for inp,target,mask in tqdm(train_loader,desc=f'v2 {epoch:03d}/{a.epochs}',dynamic_ncols=True):
          inp,target,mask=inp.to(dev,non_blocking=True),target.to(dev,non_blocking=True),mask.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True)
          with torch.autocast(device_type=dev.type,enabled=dev.type=='cuda'):proposal=model(inp)
          with torch.autocast(device_type=dev.type,enabled=False):loss=losses_v2(proposal.float()*mask.float()+inp[:,:1].float()*(1-mask.float()),target.float(),mask.float(),a.lambda_energy)[0]
          scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();hist.append(loss.item())
        sched.step();metrics,preview=evaluate(model,val_loader,dev,a.lambda_energy);row={'epoch':epoch,'train_loss':float(np.mean(hist)),**metrics,'lr':opt.param_groups[0]['lr']};writer.writerow(row);fp.flush();print(json.dumps(row),flush=True)
        state={'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'model_name':'FlowGrowthUNetV2','config':vars(a)};torch.save(state,a.output/'last.pt')
        if metrics['val_score']<best:best=metrics['val_score'];torch.save(state,a.output/'best.pt');(a.output/'best_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');save_preview(preview,a.output/'comparison.png')
    print(a.output,flush=True)
if __name__=='__main__':main()
