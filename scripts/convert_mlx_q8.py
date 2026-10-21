"""External-only affine Q8 linear weights with FP16 residual weights/activations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import mlx.core as mx
from convert_mlx_precision import digest

def convert(source,destination):
    source=source.resolve();destination=destination.resolve()
    if not str(destination).startswith('/Volumes/') or source==destination:
        raise ValueError('Use a separate external destination')
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(source.glob('*/*/graph.json')):
        spec=json.loads(path.read_text());target=destination/path.parent.relative_to(source)
        if spec.get('precision','fp32')!='fp32':raise ValueError(f'Q8 conversion requires original FP32 profiles: {path}')
        provenance={'source_graph_sha256':digest(path),'source_weights_sha256':digest(path.parent/'weights.safetensors')}
        if (target/'graph.json').exists():
            old=json.loads((target/'graph.json').read_text())
            if old.get('conversion')!=provenance or old.get('precision')!='q8-fp16-v1':raise ValueError(f'Conflicting conversion {target}')
            records.append(json.loads((target/'conversion.json').read_text()));continue
        target.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
        uses=defaultdict(list)
        def refs(value,node):
            if isinstance(value,dict):
                if 'ref' in value:uses[value['ref']].append(node)
                else:
                    for v in value.values():refs(v,node)
            elif isinstance(value,list):
                for v in value:refs(v,node)
        for node in spec['nodes']:refs(node['args'],node);refs(node['kwargs'],node)
        refs(spec['output'],{'op':'output'})
        arrays=mx.load(str(path.parent/'weights.safetensors'));converted={};quant={};total=0;packed=0
        for key,array in arrays.items():
            if not mx.issubdtype(array.dtype,mx.floating):converted[key]=array;continue
            total+=array.size
            if not bool(mx.all(mx.isfinite(array))):raise ValueError(f'Nonfinite source {key}')
            eligible=array.ndim==2 and array.shape[-1]%64==0 and uses[key] and all(
                n['op']=='aten.linear.default' and n['args'][1]=={'ref':key} for n in uses[key])
            if eligible:
                # Fit the quantizer to original FP32 values, then store scale/bias
                # in FP16 to match the packed kernel's activation precision.
                weight,scale,bias=mx.quantize(array,group_size=64,bits=8)
                converted[key]=weight
                converted[key+'__scales']=scale.astype(mx.float16)
                converted[key+'__biases']=bias.astype(mx.float16)
                quant[key]={'group_size':64,'bits':8,'shape':list(array.shape)};packed+=array.size
            else:converted[key]=array.astype(mx.float16)
        mx.eval(converted)
        for key,array in converted.items():
            if mx.issubdtype(array.dtype,mx.floating) and not bool(mx.all(mx.isfinite(array))):raise ValueError(f'FP16 overflow {key}')
        mx.save_safetensors(str(target/'weights.safetensors'),converted)
        spec.update(precision='q8-fp16-v1',conversion=provenance,quantized_weights=quant)
        record={'profile':str(path.parent.relative_to(source)),'float_parameters':total,'q8_parameters':packed,
                'q8_fraction':packed/total,'linear_layers':len(quant),'source_bytes':(path.parent/'weights.safetensors').stat().st_size,
                'converted_bytes':(target/'weights.safetensors').stat().st_size,'sha256':digest(target/'weights.safetensors'),
                'seconds':time.perf_counter()-start}
        (target/'conversion.json').write_text(json.dumps(record,indent=2)+'\n')
        (target/'graph.json').write_text(json.dumps(spec,indent=2)+'\n')
        for name in ('stats.json','provenance.json'):
            if (path.parent.parent/name).exists():shutil.copyfile(path.parent.parent/name,target.parent/name)
        records.append(record);print(json.dumps(record),flush=True)
        del arrays,converted;mx.clear_cache()
    (destination/f'conversion-{source.name}.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();convert(args.source,args.destination)
