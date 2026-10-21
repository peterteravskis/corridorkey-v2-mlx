# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Native MLX execution of source-exported CorridorKey inference graphs.

No PyTorch import or fallback: an unsupported operation fails explicitly.
FP32 is the default; converted profiles explicitly select mixed FP16 execution.
Shapes are checked against the exported profile.
"""
import json
import math
from collections import Counter
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Validated paths; FP16/Q8 use only the separately validated channel convolution.
# Benchmarks can override this before construction to retain the original path.
DEFAULT_OPTIMIZATIONS = (
    'reconstruction', 'padding', 'layout', 'dense', 'paired_dense', 'channel_conv7',
    'half_channel_conv7',
)


def execute_optimized(op, args, kwargs, optimizations):
    from .mlx_kernels import reconstruct, reflect_pad, reflect_pad_layout, reconstruct_half_experimental
    half = 'half_kernels' in optimizations
    if ('channel_conv7' in optimizations and op == 'aten.conv2d.default'
            and len(args)==7 and not kwargs and args[0].dtype==args[1].dtype==mx.float32
            and args[1].shape==(2*args[0].shape[1],1,7,7)
            and args[3]==[1,1] and args[4]==[0,0] and args[5]==[1,1]
            and args[6]==args[0].shape[1]):
        from .mlx_kernels import channel_conv7_pair
        return channel_conv7_pair(*args[:3])
    if ('dense' in optimizations and op == 'aten.linear.default' and not kwargs
            and args[0].dtype == args[1].dtype == mx.float32
            and len(args) == 3 and args[2] is not None):
        return mx.addmm(args[2], args[0], args[1].T)
    if ('reconstruction' in optimizations and op == 'aten.conv_transpose2d.input'
            and len(args) == 5 and not kwargs
            and (args[0].dtype == mx.float32 or half and args[0].dtype == mx.float16)
            and args[1].dtype == args[0].dtype and args[1].shape[2:] == (32, 32)
            and args[3] in ([16, 16], (16, 16)) and args[4] in ([8, 8], (8, 8))):
        return (reconstruct_half_experimental if args[0].dtype == mx.float16 else reconstruct)(*args[:3])
    if ('padding' in optimizations and op == 'aten.pad.default'
            and len(args) >= 3 and args[2] == 'reflect'
            and args[0].ndim == 4 and (args[0].dtype == mx.float32 or half and args[0].dtype == mx.float16)
            and len(args[1]) == 4):
        left, right, top, bottom = args[1]
        h, w = args[0].shape[-2:]
        if 0 <= min(left, right) and max(left, right) < w and 0 <= min(top, bottom) and max(top, bottom) < h:
            return (reflect_pad_layout if 'layout' in optimizations else reflect_pad)(args[0], args[1])
    return execute(op, args, kwargs)


def resize2d(x, size, align=False, cubic=False, scales=None):
    """PyTorch-compatible separable NCHW interpolation (non-antialiased)."""
    if size is None:size=[int(x.shape[i+2]*scales[i]) for i in range(2)]
    for axis,target in [(2,size[0]),(3,size[1])]:
        source=x.shape[axis]
        if source==target: continue
        scale=(source-1)/(target-1) if align and target>1 else (1/scales[axis-2] if scales else source/target)
        pos=mx.arange(target,dtype=mx.float32)*scale if align else (mx.arange(target,dtype=mx.float32)+.5)*scale-.5
        if not cubic:pos=mx.maximum(pos,0)
        base=mx.floor(pos).astype(mx.int32)
        shape=[1]*x.ndim;shape[axis]=target
        result=0
        for offset in ([-1,0,1,2] if cubic else [0,1]):
            index=base+offset
            dist=mx.abs(pos-index)
            if cubic:
                a=-.75
                w=mx.where(dist<=1,(a+2)*dist**3-(a+3)*dist**2+1,
                           mx.where(dist<2,a*dist**3-5*a*dist**2+8*a*dist-4*a,0))
            else:w=mx.maximum(1-dist,0)
            result=result+mx.take(x,mx.clip(index,0,source-1),axis=axis)*w.reshape(shape)
        x=result
    return x


def conv(x,weight,bias=None,stride=1,padding=0,dilation=1,groups=1):
    order=[0,*range(2,x.ndim),1]
    back=[0,x.ndim-1,*range(1,x.ndim-1)]
    y=mx.conv_general(x.transpose(order),weight.transpose(order),stride=stride,
                      padding=padding,kernel_dilation=dilation,groups=groups)
    if bias is not None:y=y+bias
    return y.transpose(back)


def execute(op,a,k):
    name=op.split('.')[1] if op.startswith('aten.') else op
    if name in ('view','reshape','_unsafe_view'):return mx.reshape(a[0],a[1])
    if name=='permute':return a[0].transpose(a[1])
    if name=='transpose':return mx.swapaxes(a[0],a[1],a[2])
    if name=='t':return a[0].T
    if name=='unsqueeze':return mx.expand_dims(a[0],a[1])
    if name=='squeeze':return mx.squeeze(a[0],axis=a[1] if len(a)>1 else None)
    if name=='flatten':
        x=a[0];start=a[1] if len(a)>1 else 0;end=a[2] if len(a)>2 else -1
        start%=x.ndim;end%=x.ndim
        return x.reshape((*x.shape[:start],-1,*x.shape[end+1:]))
    if name in ('clone','detach','alias','contiguous'):return a[0]
    if name=='_assert_tensor_metadata':return None
    if name=='_to_copy':return a[0].astype(k.get('dtype',a[0].dtype))
    if name=='expand':return mx.broadcast_to(a[0],[old if new==-1 else new for old,new in zip((1,)*(len(a[1])-a[0].ndim)+a[0].shape,a[1])])
    if name=='repeat':return mx.tile(a[0],a[1])
    if name=='cat':return mx.concatenate(a[0],axis=a[1] if len(a)>1 else 0)
    if name=='stack':return mx.stack(a[0],axis=a[1] if len(a)>1 else 0)
    if name=='unbind':return tuple(mx.squeeze(x,axis=a[1] if len(a)>1 else 0) for x in mx.split(a[0],a[0].shape[a[1] if len(a)>1 else 0],axis=a[1] if len(a)>1 else 0))
    if name in ('split','split_with_sizes'):
        dim=a[2] if len(a)>2 else 0
        cuts=np.cumsum(a[1])[:-1].tolist() if isinstance(a[1],list) else list(range(a[1],a[0].shape[dim],a[1]))
        return mx.split(a[0],cuts,axis=dim)
    if name=='slice':
        x,dim,start,end,*step=a;idx=[slice(None)]*x.ndim;idx[dim]=slice(start,min(end,x.shape[dim]) if end is not None else None,step[0] if step else 1)
        return x[tuple(idx)]
    if name=='select':return mx.take(a[0],a[2],axis=a[1])
    if name=='select_scatter':
        x,src,dim,index=a;idx=[slice(None)]*x.ndim;idx[dim]=index
        out=mx.array(x);out[tuple(idx)]=src;return out
    if name=='slice_scatter':
        x,src,dim,start,end,*step=a;idx=[slice(None)]*x.ndim;idx[dim]=slice(start,min(end,x.shape[dim]) if end is not None else None,step[0] if step else 1)
        out=mx.array(x);out[tuple(idx)]=src;return out
    if name=='index_select':return mx.take(a[0],a[2],axis=a[1])
    if name=='index_copy':
        x,dim,index,src=a;idx=[slice(None)]*x.ndim;idx[dim]=index
        out=mx.array(x);out[tuple(idx)]=src;return out
    if name=='index_put':
        x,indices,src=a[:3];idx=tuple(slice(None) if i is None else i for i in indices)
        if len(a)>3 and a[3]:return x.at[idx].add(src)
        out=mx.array(x);out[idx]=src;return out
    if name=='copy':return a[1].astype(a[0].dtype)
    if name=='fill':return mx.full(a[0].shape,a[1],dtype=a[0].dtype).astype(a[0].dtype)
    if name in ('logical_not','logical_and','logical_or','bitwise_not','bitwise_and','bitwise_or'):
        return getattr(mx,name)(*a)
    if name in ('new_empty','new_zeros','new_ones'):
        return (mx.ones if name=='new_ones' else mx.zeros)(a[1],dtype=k.get('dtype',a[0].dtype))
    if name in ('empty','zeros','ones','full'):
        dtype=k.get('dtype',mx.float32)
        return mx.full(a[0],a[1] if name=='full' else (1 if name=='ones' else 0),dtype=dtype).astype(dtype)
    if name in ('zeros_like','ones_like'):return getattr(mx,name)(a[0])
    if name=='arange':return mx.arange(*a,dtype=k.get('dtype',mx.int64))
    if name=='linspace':return mx.linspace(*a,dtype=k.get('dtype',mx.float32))
    if name=='scalar_tensor':return mx.array(a[0],dtype=k.get('dtype',mx.float32))
    if name=='linear':
        y=a[0]@a[1].T
        return y+a[2] if len(a)>2 and a[2] is not None else y
    if name in ('matmul','mm','bmm'):return a[0]@a[1]
    if name=='addmm':return k.get('beta',1)*a[0]+k.get('alpha',1)*(a[1]@a[2])
    if name in ('conv2d','conv3d'):return conv(*a,**k)
    if name=='conv_transpose2d':
        x,w=a[:2];bias=a[2] if len(a)>2 else None
        y=mx.conv_transpose2d(x.transpose(0,2,3,1),w.transpose(1,2,3,0),
            stride=a[3] if len(a)>3 else 1,padding=a[4] if len(a)>4 else 0,
            output_padding=a[5] if len(a)>5 else 0,groups=a[6] if len(a)>6 else 1,
            dilation=a[7] if len(a)>7 else 1)
        if bias is not None:y=y+bias
        return y.transpose(0,3,1,2)
    if name=='layer_norm':return mx.fast.layer_norm(a[0],a[2] if len(a)>2 else None,a[3] if len(a)>3 else None,a[4] if len(a)>4 else 1e-5)
    if name=='rms_norm':return mx.fast.rms_norm(a[0],a[2] if len(a)>2 else None,a[3] if len(a)>3 else 1e-6)
    if name=='scaled_dot_product_attention':
        q,key,v=a[:3];mask=a[3] if len(a)>3 else None
        if len(a)>4 and a[4]!=0:raise ValueError('Inference dropout must be zero')
        if len(a)>5 and a[5]:mask='causal'
        output=mx.fast.scaled_dot_product_attention(q,key,v,scale=k.get('scale',q.shape[-1]**-.5),mask=mask)
        # PyTorch returns zero for a fully masked query. The fused MLX kernel
        # otherwise returns a finite nonzero vector for this padded-window case.
        if isinstance(mask,mx.array):
            valid=mx.any(mask if mask.dtype==mx.bool_ else mask > -float('inf'),axis=-1,keepdims=True)
            output=mx.where(valid,output,0)
        return output
    if name=='gelu':return nn.gelu_approx(a[0]) if k.get('approximate')=='tanh' else nn.gelu(a[0])
    if name=='silu':return a[0]*mx.sigmoid(a[0])
    if name=='softmax':return mx.softmax(a[0],axis=a[1],precise=True)
    if name in ('upsample_bicubic2d','upsample_bilinear2d'):return resize2d(a[0],a[1],a[2],name=='upsample_bicubic2d',a[3] if len(a)>3 else None)
    if name=='_upsample_bilinear2d_aa':
        if tuple(a[1])==a[0].shape[-2:]:return a[0]
        raise NotImplementedError('Antialiased resizing with a changed size requires an explicit kernel')
    if name in ('constant_pad_nd','pad'):
        pads=[(0,0)]*a[0].ndim
        for i in range(len(a[1])//2):pads[-i-1]=(a[1][2*i],a[1][2*i+1])
        if name=='pad':
            mode=a[2] if len(a)>2 else 'constant'
            mode='edge' if mode=='replicate' else mode
            return mx.pad(a[0],pads,mode=mode,constant_values=(a[3] or 0) if len(a)>3 else 0)
        return mx.pad(a[0],pads,constant_values=a[2] if len(a)>2 else 0)
    if name=='roll':return mx.roll(a[0],a[1],axis=a[2])
    if name in ('clamp','clamp_min','clamp_max'):
        lo=a[1] if len(a)>1 else k.get('min');hi=a[2] if len(a)>2 else k.get('max')
        if name=='clamp_min':hi=None
        if name=='clamp_max':lo,hi=None,lo
        return mx.minimum(mx.maximum(a[0],lo) if lo is not None else a[0],hi) if hi is not None else mx.maximum(a[0],lo)
    if name in ('sum','mean','amax','amin'):
        axis=a[1] if len(a)>1 else None;keep=a[2] if len(a)>2 else False
        return getattr(mx,{'amax':'max','amin':'min'}.get(name,name))(a[0],axis=axis,keepdims=keep)
    if name=='add':return a[0]+a[1]*k.get('alpha',1)
    if name=='sub':return a[0]-a[1]*k.get('alpha',1)
    if name=='rsub':return a[1]-a[0]*(a[2] if len(a)>2 else 1)
    if name=='mul':return a[0]*a[1]
    if name=='div':return mx.floor(a[0]/a[1]) if k.get('rounding_mode')=='floor' else a[0]/a[1]
    if name=='pow':return a[0]**a[1]
    if name in ('eq','ne','lt','le','gt','ge'):
        return {'eq':mx.equal,'ne':mx.not_equal,'lt':mx.less,'le':mx.less_equal,'gt':mx.greater,'ge':mx.greater_equal}[name](a[0],a[1])
    if name=='where':return mx.where(*a)
    if name=='masked_fill':return mx.where(a[1],a[2],a[0])
    if name in ('neg','sigmoid','tanh','cos','sin','exp','sqrt','rsqrt','abs','log','floor','ceil','maximum','minimum'):
        return getattr(mx,name)(*a)
    if name=='pixel_shuffle':
        x,r=a;b,c,h,w=x.shape
        return x.reshape(b,c//(r*r),r,r,h,w).transpose(0,1,4,2,5,3).reshape(b,c//(r*r),h*r,w*r)
    if op=='<built-in function getitem>':return a[0][a[1]]
    raise NotImplementedError(f'{op}: args={[(v.shape if isinstance(v,mx.array) else v) for v in a]} kwargs={k}')


def floating_cast(value, dtype):
    if isinstance(value, mx.array):
        return value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
    if isinstance(value, list):return [floating_cast(x,dtype) for x in value]
    if isinstance(value, tuple):return tuple(floating_cast(x,dtype) for x in value)
    if isinstance(value, dict):return {k:floating_cast(v,dtype) for k,v in value.items()}
    return value


def execute_fp16(op, args, kwargs, optimizations=()):
    if ('half_channel_conv7' in optimizations and op=='aten.conv2d.default'
            and len(args)==7 and not kwargs and args[0].dtype==args[1].dtype==mx.float16
            and args[1].shape==(2*args[0].shape[1],1,7,7)
            and args[3]==[1,1] and args[4]==[0,0] and args[5]==[1,1]
            and args[6]==args[0].shape[1]):
        from .mlx_kernels import channel_conv7_pair
        return channel_conv7_pair(*args[:3])
    name=op.split('.')[1] if op.startswith('aten.') else op
    # Reduce in FP32 to avoid half-precision sum overflow and retain small
    # normalization variances. Interpolation coordinates already use FP32.
    if name in ('sum','mean','layer_norm','rms_norm','softmax','exp','pow'):
        args=floating_cast(args,mx.float32)
    result=execute_optimized(op,args,kwargs,optimizations) if 'half_kernels' in optimizations else execute(op,args,kwargs)
    # Integer indices and boolean masks must never be converted.
    return floating_cast(result,mx.float16)


class MLXGraph:
    def __init__(self,path,compile_graph=False,shared_weights=None,optimizations=None):
        path=Path(path)
        self.spec=json.loads((path/'graph.json').read_text())
        self.precision=self.spec.get('precision','fp32')
        self.optimizations = tuple(DEFAULT_OPTIMIZATIONS if optimizations is None else optimizations)
        if self.precision not in ('fp32','fp16-mixed-v1','q8-fp16-v1'):
            raise ValueError(f'Unsupported graph precision: {self.precision}')
        self.weights=mx.load(str(path/'weights.safetensors'))
        if shared_weights:
            for name,value in shared_weights.items():
                if name.startswith('p_') and name in self.weights and self.weights[name].shape==value.shape:
                    self.weights[name]=value
        if ('paired_dense' in self.optimizations and self.precision == 'fp32'
                or 'half_paired_dense' in self.optimizations and self.precision != 'fp32'):
            self._pair_dense_projections(shared_weights)
        mx.eval(self.weights)
        self.references=Counter()
        def refs(v):
            if isinstance(v,dict):
                if 'ref' in v:self.references[v['ref']]+=1
                else:
                    for x in v.values():refs(x)
            elif isinstance(v,list):
                for x in v:refs(x)
        for n in self.spec['nodes']:refs(n['args']);refs(n['kwargs'])
        refs(self.spec['output'])
        self.call=mx.compile(self._run) if compile_graph else self._run

    def _pair_dense_projections(self,shared_weights=None):
        """Pack independent gate/up projections once, without changing weights.

        Rewrite only exact shared-input linear pairs. Original tensor names are
        retained when still used elsewhere; unused originals are released.
        """
        nodes=self.spec['nodes']; replacements={}; insertions={}
        for index,node in enumerate(nodes):
            if node['name'] in replacements:continue
            if node['op']!='aten.linear.default' or len(node['args'])!=3 or node['kwargs']:continue
            ref=node['args'][1].get('ref','')
            if not ref.endswith('_mlp_w_gate_weight'):continue
            upref=ref.replace('_w_gate_weight','_w_up_weight')
            up=next((n for n in nodes[index+1:] if n['name'] not in replacements
                     and n['op']=='aten.linear.default' and len(n['args'])==3 and not n['kwargs']
                     and n['args'][1].get('ref')==upref and n['args'][0]==node['args'][0]),None)
            if up is None:continue
            if not all(isinstance(n['args'][2],dict) and 'ref' in n['args'][2] for n in [node,up]):continue
            if (node['args'][2]['ref']!=ref.removesuffix('_weight')+'_bias'
                    or up['args'][2]['ref']!=upref.removesuffix('_weight')+'_bias'):continue
            packed='_cutline_'+node['name']
            wref=ref.replace('_w_gate_weight','_paired_weight')
            bref=ref.replace('_w_gate_weight','_paired_bias')
            n=self.weights[ref].shape[0]
            if self.weights[upref].shape!=self.weights[ref].shape:continue
            quant=self.spec.get('quantized_weights',{})
            if quant.get(ref)!=quant.get(upref):continue
            if (shared_weights and wref in shared_weights and bref in shared_weights
                    and shared_weights[wref].shape==(2*n,self.weights[ref].shape[1])
                    and shared_weights[bref].shape==(2*n,)):
                self.weights[wref]=shared_weights[wref]
                self.weights[bref]=shared_weights[bref]
            else:
                self.weights[wref]=mx.concatenate([self.weights[ref],self.weights[upref]],axis=0)
                self.weights[bref]=mx.concatenate([self.weights[node['args'][2]['ref']],self.weights[up['args'][2]['ref']]])
            if ref in quant:
                quant[wref]={**quant[ref],'shape':[2*n,quant[ref]['shape'][1]]}
                for suffix in ['__scales','__biases']:
                    if shared_weights and wref+suffix in shared_weights:
                        self.weights[wref+suffix]=shared_weights[wref+suffix]
                    else:
                        self.weights[wref+suffix]=mx.concatenate([self.weights[ref+suffix],self.weights[upref+suffix]],axis=0)
            insertions[node['name']]={'name':packed,'op':'aten.linear.default',
                'args':[node['args'][0],{'ref':wref},{'ref':bref}],'kwargs':{}}
            for original,start in [(node,0),(up,n)]:
                replacements[original['name']]={'name':original['name'],'op':'aten.slice.Tensor',
                    'args':[{'ref':packed},-1,start,start+n],'kwargs':{}}
        rewritten=[]
        for node in nodes:
            if node['name'] in insertions:rewritten.append(insertions[node['name']])
            rewritten.append(replacements.get(node['name'],node))
        self.spec['nodes']=rewritten
        used=set()
        def scan(value):
            if isinstance(value,dict):
                if 'ref' in value:used.add(value['ref'])
                else:
                    for v in value.values():scan(v)
            elif isinstance(value,list):
                for v in value:scan(v)
        scan(rewritten);scan(self.spec['output'])
        for key in self.spec.get('quantized_weights',{}):
            if key in used:used.update([key+'__scales',key+'__biases'])
        self.weights={k:v for k,v in self.weights.items() if k in used}

    def _run(self,*inputs):
        env=dict(self.weights);remaining=self.references.copy()
        if self.precision!='fp32':inputs=floating_cast(inputs,mx.float16)
        for spec,value in zip(self.spec['inputs'],inputs):env[spec['name']]=value
        def decode(value):
            if isinstance(value,dict):
                if 'ref' in value:
                    name=value['ref'];result=env[name];remaining[name]-=1
                    if remaining[name]==0 and name not in self.weights:del env[name]
                    return result
                if set(value)=={'dtype'} and isinstance(value['dtype'],str):return getattr(mx,'bool_' if value['dtype']=='bool' else value['dtype'])
                if set(value)=={'device'} and isinstance(value['device'],str):return value['device']
                return {k:decode(v) for k,v in value.items()}
            if isinstance(value,list):return [decode(v) for v in value]
            return value
        for node in self.spec['nodes']:
            try:
                args=decode(node['args']);kwargs=decode(node['kwargs'])
                weight_ref=node['args'][1].get('ref') if node['op']=='aten.linear.default' else None
                quant=self.spec.get('quantized_weights',{}).get(weight_ref)
                if quant:
                    result=mx.quantized_matmul(args[0],args[1],self.weights[weight_ref+'__scales'],
                        self.weights[weight_ref+'__biases'],transpose=True,group_size=quant['group_size'],bits=8)
                    if len(args)>2 and args[2] is not None:result=result+args[2]
                    env[node['name']]=result.astype(mx.float16)
                else:
                    if self.precision != 'fp32':
                        env[node['name']] = execute_fp16(node['op'], args, kwargs, self.optimizations)
                    elif self.optimizations:
                        env[node['name']] = execute_optimized(node['op'], args, kwargs, self.optimizations)
                    else:
                        env[node['name']] = execute(node['op'], args, kwargs)
            except Exception as exc:raise RuntimeError(f"MLX node {node['name']} ({node['op']}): {exc}") from exc
        output=decode(self.spec['output'])
        # Preserve the existing CPU baton and FP32 renderer contracts.
        return floating_cast(output,mx.float32) if self.precision!='fp32' else output

    def __call__(self,*inputs):
        if len(inputs)!=len(self.spec['inputs']):raise ValueError('Wrong input count')
        for spec,value in zip(self.spec['inputs'],inputs):
            if tuple(spec['shape'])!=value.shape:raise ValueError(f"{spec['name']}: expected {spec['shape']}, got {value.shape}")
        return self.call(*inputs)
