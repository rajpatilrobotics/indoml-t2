"""Reconstruction-priority parameter gradients; official ASR is never modified."""
import math
import torch
from experiments.gap_adapt_v1.train import hybrid_loss


def soft_si_loss(pred, clean, cap_db=30.):
    """Negative centered SI-SDR with a smooth 30dB ceiling for clean examples."""
    if pred.shape != clean.shape or pred.ndim != 2:
        raise ValueError('Expected equal batch x time tensors')
    if not torch.isfinite(pred).all() or not torch.isfinite(clean).all():
        raise ValueError('Nonfinite reconstruction input')
    p=pred-pred.mean(-1,keepdim=True);c=clean-clean.mean(-1,keepdim=True)
    energy=c.square().sum(-1,keepdim=True)
    if (energy<=1e-10).any():raise ValueError('Silent clean training reference')
    target=(p*c).sum(-1,keepdim=True)/energy*c
    target_energy=target.square().sum(-1).clamp_min(1e-10)
    error=(p-target).square().sum(-1)
    return (10*torch.log10(error/target_energy+10**(-cap_db/10))).mean()


def reconstruction(pred,clean):
    return hybrid_loss(pred,clean)+.1*soft_si_loss(pred,clean)


def combine_gradients(rec, recognition, max_ratio=.5):
    """One-sided PCGrad-inspired projection and norm cap in parameter space.

    The guarantee is about raw gradients, not AdamW's preconditioned update or
    unseen evaluation metrics. Do not present this as a generalization bound.
    """
    if not 0<max_ratio<=1 or len(rec)!=len(recognition) or not rec:
        raise ValueError('Invalid gradient inputs')
    for a,b in zip(rec,recognition):
        if a.shape!=b.shape or a.device!=b.device or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite or incompatible parameter gradients')
    # Float64 scalar reductions avoid false sign/cap decisions from FP32 sums.
    rr=sum(a.double().square().sum() for a in rec)
    kk=sum(b.double().square().sum() for b in recognition)
    dot=sum((a.double()*b.double()).sum() for a,b in zip(rec,recognition))
    if not torch.isfinite(rr+kk+dot) or rr<=1e-30:
        raise ValueError('Zero or invalid reconstruction gradient')
    projection=float(torch.minimum(dot,torch.zeros_like(dot))/rr)
    adjusted=[b-a*projection for a,b in zip(rec,recognition)]
    norm2=sum(v.double().square().sum() for v in adjusted)
    scale=min(1.,max_ratio*math.sqrt(float(rr)/max(float(norm2),1e-30)))
    addition=[v*scale for v in adjusted]
    postdot=sum((a.double()*v.double()).sum() for a,v in zip(rec,addition))
    ratio=math.sqrt(float(sum(v.double().square().sum() for v in addition))/float(rr))
    if float(postdot)<-1e-5*float(rr) or ratio>max_ratio+1e-5:
        raise ValueError('Gradient preservation invariant failed')
    result=[a+v for a,v in zip(rec,addition)]
    info={'reconstruction_norm':math.sqrt(float(rr)), 'recognition_norm':math.sqrt(float(kk)),
          'gradient_cosine':float(dot/torch.sqrt(rr*kk)) if kk>0 else 0.,
          'conflict_projected':projection<0,'effective_recognition_ratio':ratio,
          'recognition_scale':scale,'raw_gradient_alignment_after':float(postdot/rr)}
    return result,info
