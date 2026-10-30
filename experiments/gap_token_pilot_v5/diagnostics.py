"""Record numerical failures; never alter gradients or continue past errors."""
import contextlib
import traceback
import warnings
from pathlib import Path
import torch
from src.data_manifest import save_json


def preserve_error(folder,error,tensors,context):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.detach().cpu() for k,v in tensors.items()},folder/'failed_tensors.pt')
    save_json(folder/'failure.json',{'status':'failed','error_type':type(error).__name__,
        'message':str(error),'traceback':traceback.format_exc(),'context':context,
        'gradients_replaced':False,'automatic_retry':False})


@contextlib.contextmanager
def failure_evidence(folder,tensors,context,anomaly=False):
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with torch.autograd.detect_anomaly(check_nan=True) if anomaly else contextlib.nullcontext():
                yield
    except Exception as error:
        preserve_error(folder,error,tensors,context)
        Path(folder,'anomaly_warnings.txt').write_text('\n'.join(str(w.message) for w in caught))
        raise


def gradient_summary(g):
    finite=bool(torch.isfinite(g).all())
    return {'all_finite':finite,'nan_count':int(torch.isnan(g).sum()),'inf_count':int(torch.isinf(g).sum()),
            'norm':float(g.double().norm()) if finite else None,'max_abs':float(g.abs().max()) if finite else None}


def repeated_gradient_check(teacher,wav,clean,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.save({'noisy':wav.detach().cpu(),'clean':clean.detach().cpu()},folder/'inputs.pt')
    report={'status':'checking','checks':[],'gradient_replacement':False};reference=None
    for repeat in range(3):
        with failure_evidence(folder,{'noisy':wav,'clean':clean},{'stage':'checkpointed_gradient','repeat':repeat},anomaly=False):
            x=wav.detach().clone().requires_grad_(True)
            loss=teacher.loss(x,clean);g,=torch.autograd.grad(loss,x)
            item={'repeat':repeat,'gradient':gradient_summary(g)}
            torch.save(g.detach().cpu(),folder/f'gradient_{repeat}.pt')
            if reference is not None and item['gradient']['all_finite']:
                item['relative_l2_vs_first']=float((g.double()-reference.double()).norm()/reference.double().norm())
                item['max_abs_difference_vs_first']=float((g-reference).abs().max())
            report['checks'].append(item);save_json(folder/'repeated_gradients.json',report)
            if not item['gradient']['all_finite']:raise ValueError('Nonfinite teacher input gradient')
            if item['gradient']['norm']<=0:raise ValueError('No posterior gradient reaches enhanced audio')
            if reference is None:reference=g.detach().clone()
    report['status']='PASS';save_json(folder/'repeated_gradients.json',report)
    print('Three original-objective gradient checks passed.',flush=True)
    return reference
