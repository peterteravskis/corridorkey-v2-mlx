"""Reduced-precision packing must retain quantization metadata and rounding."""
import json,tempfile,unittest
from pathlib import Path
import mlx.core as mx
import numpy as np
from corridorkey_mlx.mlx_graph import MLXGraph,execute_fp16
from corridorkey_mlx.mlx_kernels import channel_conv7_pair

class HalfDenseTests(unittest.TestCase):
    def test_half_convolution_rounds_before_bias(self):
        mx.random.seed(731)
        x=(mx.random.normal((2,5,11,17))*.2).astype(mx.float16)
        w=(mx.random.normal((10,1,7,7))*.1).astype(mx.float16)
        b=mx.arange(10,dtype=mx.float16)*.031
        reference=execute_fp16('aten.conv2d.default',[x,w,b,[1,1],[0,0],[1,1],5],{})
        candidate=mx.compile(channel_conv7_pair)(x,w,b)
        np.testing.assert_array_equal(np.array(reference),np.array(candidate))

    def test_packing_half_and_q8_retains_shared_arrays(self):
        mx.random.seed(732)
        for precision in ['fp16-mixed-v1','q8-fp16-v1']:
            weights={};quant={};nodes=[]
            for label in ['gate','up']:
                ref=f'p_pre_blocks_0_mlp_w_{label}_weight';bias=ref.replace('_weight','_bias')
                w=mx.random.normal((64,128))*.1
                if precision.startswith('q8'):
                    packed,s,b=mx.quantize(w,group_size=64,bits=8)
                    weights.update({ref:packed,ref+'__scales':s.astype(mx.float16),ref+'__biases':b.astype(mx.float16)})
                    quant[ref]={'group_size':64,'bits':8,'shape':[64,128]}
                else:weights[ref]=w.astype(mx.float16)
                weights[bias]=(mx.random.normal((64,))*.1).astype(mx.float16)
                nodes.append({'name':label,'op':'aten.linear.default','args':[{'ref':'x'},{'ref':ref},{'ref':bias}],'kwargs':{}})
            spec={'precision':precision,'quantized_weights':quant,'inputs':[{'name':'x','shape':[1,64,128]}],
                  'nodes':nodes,'output':[{'ref':'gate'},{'ref':'up'}]}
            with tempfile.TemporaryDirectory(prefix='cutline-half-') as directory:
                path=Path(directory);(path/'graph.json').write_text(json.dumps(spec));mx.save_safetensors(str(path/'weights.safetensors'),weights)
                old=MLXGraph(path,True,optimizations=())
                new=MLXGraph(path,True,optimizations=('half_paired_dense',))
                shared=MLXGraph(path,True,shared_weights=new.weights,optimizations=('half_paired_dense',))
                packed='p_pre_blocks_0_mlp_paired_weight'
                self.assertIs(new.weights[packed],shared.weights[packed])
                if quant:
                    self.assertEqual(new.spec['quantized_weights'][packed]['shape'],[128,128])
                    self.assertIs(new.weights[packed+'__scales'],shared.weights[packed+'__scales'])
                x=mx.random.normal((1,64,128)).astype(mx.float16)
                for a,b in zip(old(x),new(x)):np.testing.assert_array_equal(np.array(a),np.array(b))

if __name__=='__main__':unittest.main()
