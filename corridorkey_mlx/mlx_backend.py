# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""MLX neural inference with the existing, verified CPU video-I/O contract.

PyTorch is used only for shared CPU decoding, storage and window scheduling.
All five neural networks and the output renderer execute through MLX; there is no MPS or CPU
neural fallback. CPU/MLX boundary copies are included in end-to-end benchmarks.
"""
from pathlib import Path
import json
import numpy as np
import torch
import mlx.core as mx
from .mlx_graph import MLXGraph


def to_mlx(tensor):
    if isinstance(tensor,mx.array):return tensor
    if tensor.device.type!='cpu':raise ValueError('MLX adapter expects CPU I/O tensors')
    return mx.array(tensor.detach().float().numpy())

def to_torch(tensor):
    return torch.from_numpy(np.array(tensor,copy=True))

class Component(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('_cpu_io',torch.nn.Parameter(torch.zeros(1),requires_grad=False))
    def to(self,*args,**kwargs):
        if args and str(args[0]) not in ('cpu','torch.float32'):raise ValueError('MLX networks cannot be moved to a Torch device')
        return self

class RVM(Component):
    embed_dim=384
    def __init__(self,root,H,W,compile_graph,optimizations=None):
        super().__init__();self.graph=MLXGraph(root/f'{H}x{W}-t1/rvm',compile_graph,optimizations=optimizations)
    def forward(self,video,state=None):
        v=to_mlx(video)
        if state is None:carry=mx.zeros((v.shape[0],1+(v.shape[-2]//16)*(v.shape[-1]//16),384))
        else:carry=to_mlx(state)
        frames=[]
        for t in range(v.shape[1]):
            feat,carry=self.graph(v[:,t:t+1],carry)
            mx.eval(feat,carry)
            frames.append(feat)
        out=mx.concatenate(frames,axis=1);mx.eval(out)
        return to_torch(out),to_torch(carry)

class Foundations(Component):
    def __init__(self,root,H,W,compile_graph,batch_size=1,optimizations=None):
        super().__init__()
        if batch_size not in (1,2,4,8):raise ValueError('Foundation batch must be 1, 2, 4 or 8')
        self.root=root;self.H=H;self.W=W;self.compile_graph=compile_graph;self.optimizations=optimizations
        self.batch_size=batch_size;self.graphs={}
        self.graph=self.get_graph(1)
        self.rvm=RVM(root,H,W,compile_graph,optimizations)
        stats=json.loads((root/f'{H}x{W}-t1/stats.json').read_text())
        for name,value in stats.items():self.register_buffer(name,torch.tensor(value))
    def get_graph(self,T):
        if T not in self.graphs:
            shared=next(iter(self.graphs.values())).weights if self.graphs else None
            self.graphs[T]=MLXGraph(self.root/f'{self.H}x{self.W}-t{T}/foundations',self.compile_graph,shared_weights=shared,optimizations=self.optimizations)
        return self.graphs[T]
    def forward(self,rgb_cradio,rgb_moge,**kwargs):
        radio,moge=to_mlx(rgb_cradio),to_mlx(rgb_moge)
        radios=[];moges=[]
        t=0
        while t<radio.shape[0]:
            count=min(self.batch_size,2**((radio.shape[0]-t).bit_length()-1))
            r,m=self.get_graph(count)(radio[t:t+count],moge[t:t+count]);mx.eval(r,m)
            radios.append(r);moges.append(m)
            t+=count
        return {'cradio':to_torch(mx.concatenate(radios,axis=0)), 'moge':to_torch(mx.concatenate(moges,axis=0))}

class StageOne(Component):
    d_16=512
    cradio_dim=1152
    moge_dim=768
    def __init__(self,root,H,W,compile_graph,optimizations=None):
        super().__init__();self.root=root;self.H=H;self.W=W;self.compile_graph=compile_graph;self.optimizations=optimizations;self.graphs={}
        self.get_graph(1)
    def get_graph(self,T):
        if T not in self.graphs:
            shared=next(iter(self.graphs.values())).weights if self.graphs else None
            self.graphs[T]=MLXGraph(self.root/f'{self.H}x{self.W}-t{T}/stage1',self.compile_graph,shared_weights=shared,optimizations=self.optimizations)
        return self.graphs[T]
    def forward(self,rgb,hint,moge_feat,cradio_feat,rvm_feat,hint_quality,**kwargs):
        inputs=[to_mlx(x) for x in (rgb,hint,moge_feat,cradio_feat,rvm_feat,hint_quality)]
        baton,aux=self.get_graph(rgb.shape[1])(*inputs);mx.eval(baton,aux)
        return {'baton':to_torch(baton),'aux':to_torch(aux)}

class StageTwo(Component):
    def __init__(self,root,H,W,compile_graph,optimizations=None):
        super().__init__(); self.root=root; self.H=H; self.W=W; self.compile_graph=compile_graph; self.optimizations=optimizations; self.graphs={}
        self.get_graph(1)
    def get_graph(self,T):
        if T not in self.graphs:
            shared=next(iter(self.graphs.values())).weights if self.graphs else None
            self.graphs[T]=MLXGraph(self.root/f'{self.H}x{self.W}-t{T}/stage2',self.compile_graph,shared_weights=shared,optimizations=self.optimizations)
        return self.graphs[T]
    def forward(self,rgb,baton):
        # Power-of-two graphs cover every remainder without padding or changing frames.
        outputs=[]; start=0; T=rgb.shape[1]
        while start<T:
            count=2**((T-start).bit_length()-1)
            result=self.get_graph(count)(to_mlx(rgb[:,start:start+count]),to_mlx(baton[:,start:start+count]))
            mx.eval(result); outputs.append(result[0]); start+=count
        return mx.concatenate(outputs,axis=1) if len(outputs)>1 else outputs[0]

class MLXModel(Component):
    def __init__(self,root='.mlx-models',height=1024,width=1536,compile_graph=False,cache_mib=256,foundation_batch=1,optimizations=None):
        super().__init__();root=Path(root)
        # Do not let the allocator retain gigabytes of dead activations between
        # frames on a shared-memory desktop. Live tensors are unaffected.
        mx.set_cache_limit(cache_mib*1024*1024)
        self._inference_device=torch.device('cpu')
        self._inference_backend='mlx'
        self.foundations=Foundations(root,height,width,compile_graph,foundation_batch,optimizations)
        self.stage1=StageOne(root,height,width,compile_graph,optimizations)
        self.stage2=StageTwo(root,height,width,compile_graph,optimizations)
        from .mlx_render import MLXRenderer
        self._render_predictions=MLXRenderer()
        self.eval()
