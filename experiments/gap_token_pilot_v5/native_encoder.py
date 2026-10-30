"""Scoped native encoder derivatives; official transcription is not wrapped."""
import torch
from experiments.gap_asr_v1.teacher import features,finite


def backward_nodes(tensor):
    stack=[tensor.grad_fn];seen=set();names=set()
    while stack:
        node=stack.pop()
        if node is None or node in seen:continue
        seen.add(node);names.add(type(node).__name__)
        stack.extend(n for n,_ in node.next_functions)
    return sorted(names)


def encode_native(teacher,wav):
    f,lengths=features(teacher.model,wav)
    # PyTorch 2.6 _select_batch_norm_backend consults the global cuDNN flag.
    # Disable JIT optimized execution too, so a cached graph cannot specialize
    # this call around the previous cuDNN choice. No weights or graph edits.
    with torch.jit.optimized_execution(False), torch.backends.cudnn.flags(enabled=False):
        e,lens=teacher.model.encoder(f.to(teacher.model._io_dtype),lengths)
    names=backward_nodes(e) if e.requires_grad else []
    if any('CudnnBatchNormBackward' in n for n in names):
        raise RuntimeError('Native encoder repair still reached cuDNN BatchNorm backward')
    if names:
        teacher.encoder_backend_evidence={'policy':'native_encoder_no_jit_optimization',
            'backward_node_types':names,'offending_cudnn_batchnorm_node_absent':True,
            'official_transcription_backend_changed':False}
    return finite(e,'encoder output'),lens
