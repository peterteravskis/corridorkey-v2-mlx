# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Render the current CLI benchmark receipts as charcoal-background figures."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from render_benchmarks import COLORS, BG, PANEL, INK, MUTED, header, style, save

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'benchmarks/cli/summary.json').read_text())
assert D['status']=='complete' and len(D['conditions'])==30
PRECISIONS=['fp32','fp16','q8']
CASES=['fur-motion','blue-hair-gauze','hair-gauze','blue-fur-glass','moving-32']
LABELS=['Squirrel · green','Redhead / gauze · blue','Hair / gauze · green','Fur / glass · blue','Moving video · 32 frames']

def row(case,precision,mode='enabled'):
    return next(r for r in D['conditions'] if (r['case'],r['precision'],r['kernel_mode'])==(case,precision,mode))

def legend(fig,ax):
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.125),ncol=3)

def workflow():
    fig,axes=plt.subplots(1,2,figsize=(14,8.5))
    fig.subplots_adjust(left=.22,right=.95,bottom=.27,top=.75,wspace=.25)
    header(fig,'CLI workflow time and physical memory','M4 Pro, 48 GiB · enabled kernels · one complete warm-up and three measured runs per condition',
        'Actual CLI: decoding, neural inference, rendering, four 16-bit PNG writes, and scratch cleanup. Model loading and warm-up excluded.\n'
        'Stills: 1536 × 1024. Video: native 1280 × 720, 32 moving frames; seconds divided by 32. Memory: sampled process physical footprint.')
    for j,p in enumerate(PRECISIONS):
        y=np.arange(5)+(j-1)*.24
        values=[row(c,p)['median_seconds']/row(c,p)['frames'] for c in CASES]
        memory=[row(c,p)['peak_physical_bytes']/2**30 for c in CASES]
        for ax,v in zip(axes,(values,memory)):ax.barh(y,v,height=.2,color=COLORS[j],label=p.upper())
    for i,ax in enumerate(axes):
        ax.set_yticks(np.arange(5),LABELS if i==0 else ['']*5)
        ax.invert_yaxis();style(ax,'seconds / frame' if i==0 else 'GiB · warmed peak physical footprint')
        ax.set_title('Completed CLI workflow' if i==0 else 'Physical memory')
    legend(fig,axes[0]);save(fig,'cli-workflow')

def gains():
    fig,ax=plt.subplots(figsize=(14,7.5));fig.subplots_adjust(left=.12,right=.78,bottom=.27,top=.75)
    header(fig,'CLI workflow with and without kernel optimizations','32 moving frames · identical precision, input, temporal settings and completed outputs',
        'Reference disables custom graph rewrites and kernels; enabled uses the release defaults. Bars: median; dots: all three measured runs.\n'
        'One full warm-up per process. Includes decoding through completed PNG writing and scratch cleanup. Initial load and compilation excluded.')
    maximum=max(x['seconds'] for p in PRECISIONS for m in ['reference','enabled'] for x in row('moving-32',p,m)['runs'])
    for i,p in enumerate(PRECISIONS):
        for j,m in enumerate(['reference','enabled']):
            r=row('moving-32',p,m);y=i+j*.35;v=r['median_seconds']
            ax.barh(y,v,height=.29,color=COLORS[j],label=m.capitalize() if i==0 else None)
            ax.scatter([x['seconds'] for x in r['runs']],[y]*3,color=INK,s=16,edgecolors=BG,linewidths=.4,zorder=4)
            ax.text(max(x['seconds'] for x in r['runs'])+maximum*.02,y,f'{v:.2f} s',va='center',fontsize=10)
        change=100*(1-row('moving-32',p)['median_seconds']/row('moving-32',p,'reference')['median_seconds'])
        ax.text(maximum*1.29,i+.12,f'{abs(change):.1f}%',color=COLORS[1],fontsize=23,weight='bold',va='center')
        ax.text(maximum*1.29,i+.39,'less time' if change>=0 else 'more time',color=MUTED,fontsize=10,va='center')
    ax.set_xlim(0,maximum*1.25);ax.set_yticks(np.arange(3)+.175,[p.upper() for p in PRECISIONS]);ax.invert_yaxis()
    style(ax,'seconds / 32-frame CLI workflow');legend(fig,ax);save(fig,'cli-kernel-gains')

def stages():
    fig,ax=plt.subplots(figsize=(14,8));fig.subplots_adjust(left=.2,right=.96,bottom=.31,top=.75)
    header(fig,'CLI time by processing stage','32 moving frames · reference and enabled kernels · same input and precision in each pair',
        'Breakdown uses the median-duration measured run in each condition. Calls include their CPU/MLX tensor bridges; rendering is evaluated.\n'
        'Other includes decoding, preparation, feature staging, temporal blending, scheduling, and scratch cleanup. All output writes are included.')
    keys=['rvm','foundations','stage1','stage2','render','write','other']
    names=['RVM','C-RADIO + MoGe','Matting stage 1','Matting stage 2','Rendering','PNG writes','Other I/O / CPU']
    colors=['#a2b2cb','#8299c1','#9a8ab9','#67dbba','#c3cf8b','#ecb875','#68717e']
    labels=[]
    for i,(p,m) in enumerate((p,m) for p in PRECISIONS for m in ['reference','enabled']):
        r=row('moving-32',p,m);sample=sorted(r['runs'],key=lambda x:x['seconds'])[1]
        values=dict(sample['stage_seconds']);values['other']=max(0,sample['seconds']-sum(values.values()))
        left=0
        for key,color,name in zip(keys,colors,names):
            ax.barh(i,values.get(key,0),left=left,height=.58,color=color,label=name if i==0 else None)
            left+=values.get(key,0)
        labels.append(f'{p.upper()} · {m}')
    ax.set_yticks(range(6),labels);ax.invert_yaxis();style(ax,'seconds / 32-frame CLI workflow')
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.16),ncol=4,fontsize=10)
    save(fig,'cli-stage-times')

def startup():
    fig,ax=plt.subplots(figsize=(14,7.5));fig.subplots_adjust(left=.12,right=.94,bottom=.27,top=.75)
    header(fig,'Initial model load and first CLI workflow','32 moving frames · enabled kernels · first workflow includes graph compilation',
        'First use is initial model construction plus the first complete workflow; Python interpreter/import startup is excluded.\n'
        'Warmed comparison is the median of the next three runs in the same process. Every workflow writes all four PNG outputs.')
    for i,p in enumerate(PRECISIONS):
        r=row('moving-32',p);load=r['model_load_seconds'];first=r['warmups'][0]['seconds'];warm=r['median_seconds']
        ax.barh(i,load,height=.27,color=COLORS[0],label='Initial model load' if i==0 else None)
        ax.barh(i,first,left=load,height=.27,color=COLORS[2],label='First workflow / compile' if i==0 else None)
        ax.barh(i+.34,warm,height=.27,color=COLORS[1],label='Warmed workflow' if i==0 else None)
        ax.text(load+first+1,i,f'{load+first:.1f} s',va='center');ax.text(warm+1,i+.34,f'{warm:.1f} s',va='center')
    ax.set_yticks(np.arange(3)+.17,[p.upper() for p in PRECISIONS]);ax.invert_yaxis();style(ax,'seconds')
    ax.set_xlim(right=ax.get_xlim()[1]*1.12);legend(fig,ax);save(fig,'cli-first-use')

def agreement():
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(1,2,figsize=(14,8.5));fig.subplots_adjust(left=.23,right=.94,bottom=.27,top=.74,wspace=.22)
    header(fig,'Output agreement after enabling kernel optimizations','Maximum absolute 16-bit code difference · each precision compared with its own reference path',
        'All pixels and all frames compared for alpha and checker outputs; values are differences in the saved PNGs, not perceptual scores.\n'
        'This does not compare FP16/Q8 against FP32. Visible foreground and RGBA checks, limits, and per-file hashes are in the raw receipts.')
    for ax,output in zip(axes,['alpha','checker']):
        values=np.zeros((5,3))
        for i,c in enumerate(CASES):
            for j,p in enumerate(PRECISIONS):
                data=json.loads((ROOT/f'benchmarks/cli/{c}-{p}-quality.json').read_text())
                values[i,j]=round(max(r['max'] for r in data['rows'] if r['file'].startswith(output+'/'))*65535)
        ax.imshow(values>0,cmap=ListedColormap(['#28574f','#735c39']),vmin=0,vmax=1,aspect='auto')
        for i in range(5):
            for j in range(3):ax.text(j,i,str(int(values[i,j])),ha='center',va='center',fontsize=19,color=INK)
        ax.set_xticks(range(3),[p.upper() for p in PRECISIONS]);ax.set_yticks(range(5),LABELS if output=='alpha' else ['']*5)
        ax.tick_params(length=0,pad=12);ax.set_title('Alpha' if output=='alpha' else 'Checker composite')
        for spine in ax.spines.values():spine.set_visible(False)
    fig.legend(handles=[Patch(color='#28574f',label='Exact pixels'),Patch(color='#735c39',label='Nonzero numerical difference')],
               loc='center',bbox_to_anchor=(.5,.125),ncol=2)
    save(fig,'cli-output-agreement')

if __name__=='__main__':
    workflow();gains();stages();startup();agreement()
