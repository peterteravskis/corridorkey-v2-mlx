# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""MLX colour decode, screen despill and straight-alpha export; green matches MPS."""
import math
import numpy as np
import mlx.core as mx

def linear_to_srgb(x):
    return mx.where(x<=.0031308,x*12.92,1.055*mx.maximum(x,.0031308)**(1/2.4)-.055)

def srgb_to_linear(x):
    return mx.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

class MLXRenderer:
    def __init__(self):self.checkers={};self.functions={}
    def __call__(self,meta,outputs,checker,logits):
        if tuple(meta['model_native_hw'])!=tuple(meta['output_hw']):raise ValueError('Native model scale required')
        H,W=meta['output_hw'];left,_,top,_=meta['pad']
        strength=float(meta.get('despill_strength',1.))
        screen=meta.get('screen','green')
        if screen not in ('green','blue'):raise ValueError('Screen must be green or blue')
        if not 0<=strength<=1:raise ValueError('Despill strength must be in [0,1]')
        key=(H,W,left,top,tuple(sorted(outputs)),strength,bool(meta.get('cutout_linear',False)),
             float(meta.get('linear_brightness',1.)),float(meta.get('linear_contrast',1.)),float(meta.get('linear_contrast_pivot',.18)),screen)
        if (H,W) not in self.checkers:self.checkers[H,W]=mx.array(checker.numpy())[None]
        if key not in self.functions:
            def render(x,bg):
                x=x[...,top:top+H,left:left+W]
                alpha=mx.clip(x[:,:1],0,1)
                linear=mx.sinh(x[:,1:4]*math.asinh(16/.008))*.008
                if abs(key[7]-1)>1e-6 or abs(key[8]-1)>1e-6:linear=(linear/key[7]-key[9])/key[8]+key[9]
                raw=mx.clip(linear_to_srgb(mx.maximum(linear,0)),0,1)
                channel=1 if screen=='green' else 2
                others=[c for c in range(3) if c!=channel]
                spill=mx.maximum(raw[:,channel:channel+1]-(raw[:,others[0]:others[0]+1]+raw[:,others[1]:others[1]+1])*.5,0)*strength
                neutral=mx.concatenate([raw[:,c:c+1]-spill if c==channel else raw[:,c:c+1] for c in range(3)],axis=1)
                clean=mx.clip(neutral+mx.minimum((.7152 if channel==1 else .0722)*spill,mx.maximum(1-mx.max(neutral,axis=1,keepdims=True),0)),0,1)
                result=[]
                for name in key[4]:
                    if name=='alpha':rgb=mx.broadcast_to(alpha,(alpha.shape[0],3,H,W))
                    elif name=='raw_fg':rgb=raw
                    elif name=='fg':rgb=clean
                    elif name=='checker':rgb=mx.clip(linear_to_srgb(mx.maximum(srgb_to_linear(clean)*alpha+srgb_to_linear(bg)*(1-alpha),0)),0,1)
                    elif name=='cutout':rgb=srgb_to_linear(clean) if key[6] else clean
                    else:raise ValueError(name)
                    bgra=rgb[:,[2,1,0]]
                    if name=='cutout':bgra=mx.concatenate([bgra,alpha],axis=1)
                    result.append((mx.clip(bgra,0,1)*65535).transpose(0,2,3,1).astype(mx.uint16))
                return result,mx.all(mx.isfinite(x))
            self.functions[key]=mx.compile(render)
        arrays,finite=self.functions[key](logits,self.checkers[H,W]);mx.eval(arrays,finite)
        if not bool(finite):raise FloatingPointError('Non-finite MLX predictions')
        return {name:np.array(array) for name,array in zip(key[4],arrays)}
