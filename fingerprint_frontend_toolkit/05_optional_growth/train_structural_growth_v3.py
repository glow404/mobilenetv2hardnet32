#!/usr/bin/env python3
"""Train V3: mask-aware dual-stream fingerprint ridge reconstruction.

Outputs are ridge intensity, ridge orientation (sin/cos 2theta), ridge
frequency, and reconstruction confidence.  The texture and structural streams
are fused before decoding.  A conditional PatchGAN is enabled only after a
warm-up so it sharpens already geometry-constrained ridge predictions instead
of inventing global texture.
"""
from __future__ import annotations

import argparse, csv, json, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_ridge_gap_growth import make_growth_mask, tensor_fields, gabor_filters, grad_xy, dilate, save_preview
from train_realistic_growth_v2 import realistic_degrade, read_gray


FREQ_MIN, FREQ_MAX = 1 / 14.0, 1 / 3.5


def estimate_frequency(gray: np.ndarray, theta: np.ndarray, block: int = 16) -> np.ndarray:
    """Estimate local ridge frequency from normal-direction profiles.

    This is deliberately a label/conditioning estimate, not a final enhancer.
    Frequency is evaluated only on clean training targets; V3 learns to infer
    it from the visible mask boundary at inference time.
    """
    h, w = gray.shape; output = np.full((h, w), 1 / 8.0, np.float32)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    offsets = np.arange(-16, 16, dtype=np.float32)
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            y1, x1 = min(y0 + block, h), min(x0 + block, w)
            cy, cx = (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2
            # Ridge normal is tangent - pi/2; averaging several parallel
            # profiles makes frequency robust to local brightness variations.
            normal = float(theta[int(round(cy)), int(round(cx))] - np.pi / 2)
            values = []
            tangent = theta[int(round(cy)), int(round(cx))]
            for side in (-6.0, 0.0, 6.0):
                mx = (cx + side * np.cos(tangent) + offsets * np.cos(normal)).astype(np.float32)
                my = (cy + side * np.sin(tangent) + offsets * np.sin(normal)).astype(np.float32)
                values.append(cv2.remap(gray, mx[None], my[None], cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)[0])
            profile = np.mean(values, axis=0); profile -= profile.mean()
            spectrum = np.abs(np.fft.rfft(profile))
            freqs = np.fft.rfftfreq(profile.size)
            allowed = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
            frequency = float(freqs[allowed][np.argmax(spectrum[allowed])]) if allowed.any() else 1 / 8.0
            output[y0:y1, x0:x1] = frequency
    return output


class StructuralDataset(Dataset):
    def __init__(self, files: list[Path], real_masks: list[Path], train: bool, seed: int):
        self.files, self.real_masks, self.train, self.seed = files, real_masks, train, seed

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        target = read_gray(self.files[index]).astype(np.float32) / 255.0
        rng = np.random.default_rng(self.seed + index + (random.randrange(1 << 20) if self.train else 0))
        if self.train and rng.random() < .5: target = np.fliplr(target).copy()
        if self.train and rng.random() < .5: target = np.flipud(target).copy()
        if self.real_masks and rng.random() < .75:
            raw = read_gray(self.real_masks[int(rng.integers(len(self.real_masks)))])
            mask = cv2.resize((raw > 0).astype(np.uint8), target.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)
            if self.train and rng.random() < .5: mask = np.fliplr(mask).copy()
            if self.train and rng.random() < .5: mask = np.flipud(mask).copy()
            if not .01 < mask.mean() < .30: mask = make_growth_mask(target, rng)
        else: mask = make_growth_mask(target, rng)
        corrupt = realistic_degrade(target, mask, rng)
        theta, coherence, _ = tensor_fields(target, np.zeros_like(target))
        frequency = estimate_frequency(target, theta)
        a2 = 2 * theta
        # Visible-field input must be calculated from the corrupted image.
        visible_theta, _, visible_conf = tensor_fields(corrupt, mask)
        v2 = 2 * visible_theta
        texture = np.stack([corrupt, mask]).astype(np.float32)
        structure = np.stack([np.sin(v2), np.cos(v2), visible_conf, mask]).astype(np.float32)
        labels = np.stack([target, np.sin(a2), np.cos(a2), frequency, coherence]).astype(np.float32)
        return torch.from_numpy(texture), torch.from_numpy(structure), torch.from_numpy(labels), torch.from_numpy(mask[None])


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); groups = next(group for group in range(min(8, co), 0, -1) if co % group == 0)
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(groups, co), nn.SiLU(),
                                 nn.Conv2d(co, co, 3, padding=1, bias=False), nn.GroupNorm(groups, co), nn.SiLU())
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(self, x): return self.net(x) + self.skip(x)


class DualStreamGrowthNet(nn.Module):
    """Texture stream + orientation/frequency-prior stream, fused at all scales."""
    def __init__(self, base: int = 28):
        super().__init__()
        self.te1,self.se1=ConvBlock(2,base),ConvBlock(4,base)
        self.te2,self.se2=ConvBlock(base,2*base),ConvBlock(base,2*base)
        self.te3,self.se3=ConvBlock(2*base,4*base),ConvBlock(2*base,4*base)
        self.mid=ConvBlock(8*base,8*base)
        self.d3=ConvBlock(16*base,4*base); self.d2=ConvBlock(6*base,2*base); self.d1=ConvBlock(3*base,base)
        self.head=nn.Conv2d(base,5,1)
    @staticmethod
    def up(x, ref): return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    def forward(self, texture, structure):
        t1,s1=self.te1(texture),self.se1(structure)
        t2,s2=self.te2(F.max_pool2d(t1,2)),self.se2(F.max_pool2d(s1,2))
        t3,s3=self.te3(F.max_pool2d(t2,2)),self.se3(F.max_pool2d(s2,2))
        z=self.mid(torch.cat([F.max_pool2d(t3,2),F.max_pool2d(s3,2)],1))
        z=self.d3(torch.cat([self.up(z,t3),t3,s3],1))
        z=self.d2(torch.cat([self.up(z,t2),t2],1))
        z=self.d1(torch.cat([self.up(z,t1),t1],1))
        raw=self.head(z)
        ridge=torch.sigmoid(raw[:,:1]); orient=F.normalize(torch.tanh(raw[:,1:3]),dim=1,eps=1e-6)
        frequency=FREQ_MIN+(FREQ_MAX-FREQ_MIN)*torch.sigmoid(raw[:,3:4]); confidence=torch.sigmoid(raw[:,4:5])
        return torch.cat([ridge,orient,frequency,confidence],1)


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 7):
        super().__init__(); layers=[]; widths=(48,96,192,256); ci=channels
        for i,co in enumerate(widths):
            layers += [nn.Conv2d(ci,co,4,stride=2 if i<3 else 1,padding=1), nn.LeakyReLU(.2,True)]
            ci=co
        layers.append(nn.Conv2d(ci,1,3,padding=1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def structural_loss(pred, labels, mask):
    support=dilate(mask,3); ridge,psin,pcos,pfreq,pconf=pred.split(1,1); target,tsin,tcos,tfreq,tconf=labels.split(1,1)
    l1=((ridge-target).abs()*mask).sum()/(mask.sum()+1e-6)
    orient=(1-(psin*tsin+pcos*tcos).clamp(-1,1)); orient=(orient*support).sum()/(support.sum()+1e-6)
    freq=((pfreq-tfreq).abs()*support).sum()/(support.sum()+1e-6)
    confidence=F.binary_cross_entropy(pconf.clamp(1e-4,1-1e-4), (tconf>.28).float(), reduction='none'); confidence=(confidence*support).sum()/(support.sum()+1e-6)
    px,py=grad_xy(ridge);tx,ty=grad_xy(target); grad=(((px-tx).abs()+(py-ty).abs())*support).sum()/(2*support.sum()+1e-6)
    bank=gabor_filters(ridge.device,torch.float32); pg,tg=F.conv2d(ridge,bank,padding=7),F.conv2d(target,bank,padding=7)
    gabor=((pg-tg).abs()*support).sum()/(support.sum()*bank.shape[0]+1e-6)
    total=l1+.35*orient+.35*freq+.10*confidence+.25*grad+2.0*gabor
    return total,{'ridge_l1':l1,'orientation':orient,'frequency':freq,'confidence':confidence,'gradient':grad,'gabor':gabor}


def conditional_input(texture, maps): return torch.cat([texture, maps],1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); values=[]; previews=[]
    for texture,structure,labels,mask in loader:
        texture,structure,labels,mask=[x.to(device) for x in (texture,structure,labels,mask)]
        pred=model(texture,structure); result=pred[:,:1]*mask+texture[:,:1]*(1-mask)
        loss,parts=structural_loss(torch.cat([result,pred[:,1:]],1),labels,mask)
        mse=(((result-labels[:,:1]).square()*mask).sum()/(mask.sum()+1e-6)).item()
        row={'score':loss.item(),'psnr':-10*np.log10(max(mse,1e-12)),**{k:v.item() for k,v in parts.items()}};values.append(row)
        if len(previews)<8:
            for i in range(min(texture.shape[0],8-len(previews))): previews.append([labels[i,0].cpu().numpy(),texture[i,0].cpu().numpy(),result[i,0].cpu().numpy(),mask[i,0].cpu().numpy()])
    return {k:float(np.mean([x[k] for x in values])) for k in values[0]},previews


def main():
    p=argparse.ArgumentParser();p.add_argument('--clean-root',type=Path,default=Path('指纹识别数据集/不贴屏不贴膜/butieping'));p.add_argument('--mask-root',type=Path,default=Path('pipeline_outputs/growth_then_gan_workset_30_20260929/images'));p.add_argument('--output',type=Path,default=None);p.add_argument('--init',type=Path,default=None,help='optional compatible generator checkpoint for domain fine-tuning');p.add_argument('--epochs',type=int,default=100);p.add_argument('--warmup-epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--workers',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--gan-weight',type=float,default=.04);p.add_argument('--max-train',type=int,default=0);p.add_argument('--max-val',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260930);a=p.parse_args()
    root=Path.cwd()
    for n in ('clean_root','mask_root','init'):
        value=getattr(a,n)
        if value is not None and not value.is_absolute():setattr(a,n,root/value)
    if a.output is None:a.output=root/'fingergrowthV5'/'experiments'/f'structural_growth_v3_{time.strftime("%Y%m%d_%H%M%S")}'
    elif not a.output.is_absolute():a.output=root/a.output
    if a.output.exists():raise FileExistsError(a.output)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    files=sorted(a.clean_root.rglob('*.bmp'));groups={}
    # Both supported sources are hierarchical, but not identically so:
    # V7 is ``root/<identity>/pair.bmp`` while the ultrasound corpus has an
    # additional collection layer.  The first relative component is the only
    # unambiguous identity split for the V7 adaptation experiment.
    for f in files:
        rel=f.relative_to(a.clean_root)
        groups.setdefault(rel.parts[0],[]).append(f)
    ids=sorted(groups);val_ids=set(ids[-max(2,round(.2*len(ids))):]);train=[f for k in ids if k not in val_ids for f in groups[k]];val=[f for k in ids if k in val_ids for f in groups[k]]
    random.Random(a.seed).shuffle(train);random.Random(a.seed+1).shuffle(val)
    if a.max_train:train=train[:a.max_train]
    if a.max_val:val=val[:a.max_val]
    masks=sorted(a.mask_root.rglob('mask.png'))
    # A missing harvested-real-mask pool is not a training failure: the
    # dataset falls back to procedurally generated gaps.  This keeps a domain
    # adaptation pilot reproducible after old visual worksets are cleaned.
    device=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu');a.output.mkdir(parents=True)
    (a.output/'split.json').write_text(json.dumps({'train':len(train),'val':len(val),'val_ids':sorted(val_ids),'masks':len(masks),'config':vars(a)},indent=2,default=str),encoding='utf-8')
    loader=lambda ds,sh:DataLoader(ds,batch_size=a.batch_size,shuffle=sh,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    tr,va=loader(StructuralDataset(train,masks,True,a.seed),True),loader(StructuralDataset(val,masks,False,a.seed+100000),False)
    model=DualStreamGrowthNet().to(device)
    if a.init is not None:
        state=torch.load(a.init,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
    disc=PatchDiscriminator().to(device);optg=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.5,.999),weight_decay=1e-4);optd=torch.optim.AdamW(disc.parameters(),lr=a.lr,betas=(.5,.999));sched=torch.optim.lr_scheduler.CosineAnnealingLR(optg,T_max=a.epochs);best=float('inf')
    fields=['epoch','train_g','train_d','val_score','val_psnr','val_ridge_l1','val_orientation','val_frequency','val_gabor','lr']
    with (a.output/'train.csv').open('w',newline='',encoding='utf-8') as fp:
      writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
      for epoch in range(1,a.epochs+1):
        model.train();disc.train();gl,dl=[],[];gan_on=epoch>a.warmup_epochs
        for texture,structure,labels,mask in tqdm(tr,desc=f'v3 {epoch:03d}/{a.epochs}',dynamic_ncols=True):
          texture,structure,labels,mask=[x.to(device,non_blocking=True) for x in (texture,structure,labels,mask)]
          pred=model(texture,structure);fake=torch.cat([pred[:,:1]*mask+texture[:,:1]*(1-mask),pred[:,1:]],1);base,_=structural_loss(fake,labels,mask)
          if gan_on:
            optd.zero_grad(set_to_none=True);real=conditional_input(texture,labels);fake_cond=conditional_input(texture,fake.detach());d_loss=F.softplus(-disc(real)).mean()+F.softplus(disc(fake_cond)).mean();d_loss.backward();optd.step();dl.append(d_loss.item())
            gan=F.softplus(-disc(conditional_input(texture,fake))).mean();loss=base+a.gan_weight*gan
          else: loss=base;dl.append(0.0)
          optg.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optg.step();gl.append(loss.item())
        sched.step();metrics,preview=evaluate(model,va,device);row={'epoch':epoch,'train_g':float(np.mean(gl)),'train_d':float(np.mean(dl)),'val_score':metrics['score'],'val_psnr':metrics['psnr'],'val_ridge_l1':metrics['ridge_l1'],'val_orientation':metrics['orientation'],'val_frequency':metrics['frequency'],'val_gabor':metrics['gabor'],'lr':optg.param_groups[0]['lr']};writer.writerow(row);fp.flush();print(json.dumps(row),flush=True)
        state={'model':model.state_dict(),'discriminator':disc.state_dict(),'epoch':epoch,'metrics':metrics,'model_name':'DualStreamGrowthNet','config':vars(a)};torch.save(state,a.output/'last.pt')
        if metrics['score']<best:best=metrics['score'];torch.save(state,a.output/'best.pt');(a.output/'best_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');save_preview(preview,a.output/'comparison.png')
    print(a.output,flush=True)
if __name__=='__main__':main()
