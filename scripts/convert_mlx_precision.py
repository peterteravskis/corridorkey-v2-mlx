"""Create opt-in FP16 MLX profiles on external storage; never alter the reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import mlx.core as mx


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve();destination=args.destination.resolve()
    if not str(destination).startswith('/Volumes/') or destination==source:
        raise ValueError('Conversion output must be a separate external-volume directory')
    destination.mkdir(parents=True,exist_ok=True)
    report={'source':str(source),'precision':'fp16-mixed-v1','profiles':[]}
    shared={}
    for path in sorted(source.glob('*/*/graph.json')):
        relative=path.parent.relative_to(source);target=destination/relative
        target.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();weights=path.parent/'weights.safetensors'
        spec=json.loads(path.read_text());spec['precision']='fp16-mixed-v1'
        spec['conversion']={'source_graph_sha256':digest(path),'source_weights_sha256':digest(weights)}
        identity=(weights.stat().st_dev,weights.stat().st_ino)
        if (target/'graph.json').exists():
            previous=json.loads((target/'graph.json').read_text())
            if previous.get('conversion')!=spec['conversion'] or previous.get('precision')!=spec['precision']:
                raise ValueError(f'Existing conversion has different provenance: {target}')
            if not (target/'weights.safetensors').exists():raise ValueError(f'Incomplete conversion: {target}')
            shared[identity]=target/'weights.safetensors'
        elif identity in shared:
            try:os.link(shared[identity],target/'weights.safetensors')
            except OSError:shutil.copyfile(shared[identity],target/'weights.safetensors')
        else:
            arrays=mx.load(str(weights));converted={}
            for key,array in arrays.items():
                if mx.issubdtype(array.dtype,mx.floating):
                    if not bool(mx.all(mx.isfinite(array))):raise ValueError(f'Nonfinite source: {relative}/{key}')
                    cast=array.astype(mx.float16);mx.eval(cast)
                    if not bool(mx.all(mx.isfinite(cast))):raise ValueError(f'FP16 overflow: {relative}/{key}')
                    converted[key]=cast
                else:converted[key]=array
            mx.save_safetensors(str(target/'weights.safetensors'),converted)
            del arrays,converted
            mx.clear_cache()
            shared[identity]=target/'weights.safetensors'
        (target/'graph.json').write_text(json.dumps(spec,indent=2)+'\n')
        for name in ('stats.json','provenance.json'):
            if (path.parent.parent/name).exists():shutil.copyfile(path.parent.parent/name,target.parent/name)
        record={'profile':str(relative),'source_bytes':weights.stat().st_size,
                'converted_bytes':(target/'weights.safetensors').stat().st_size,
                'sha256':digest(target/'weights.safetensors'),'seconds':time.perf_counter()-start}
        report['profiles'].append(record);print(json.dumps(record),flush=True)
    (destination/f'conversion-{source.name}.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
