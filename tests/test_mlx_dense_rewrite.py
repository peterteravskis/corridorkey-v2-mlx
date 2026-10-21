"""Gate/up packing must preserve shared weights, inputs and graph outputs."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import mlx.core as mx
from corridorkey_mlx.mlx_graph import MLXGraph


class DenseRewriteTests(unittest.TestCase):
    def test_shared_gate_weight_survives_pairing(self):
        mx.random.seed(725)
        gate='p_pre_blocks_0_mlp_w_gate_weight';up='p_pre_blocks_0_mlp_w_up_weight'
        gb=gate.replace('_weight','_bias');ub=up.replace('_weight','_bias')
        weights={gate:mx.random.normal((16,8))*.1,up:mx.random.normal((16,8))*.1,
                 gb:mx.arange(16,dtype=mx.float32)*.01,ub:mx.arange(16,dtype=mx.float32)*-.01}
        def linear(name,w,b):return {'name':name,'op':'aten.linear.default','args':[{'ref':'x'},{'ref':w},{'ref':b}],'kwargs':{}}
        nodes=[linear('gate',gate,gb),{'name':'activation','op':'aten.silu.default','args':[{'ref':'gate'}],'kwargs':{}},
               linear('up',up,ub),{'name':'product','op':'aten.mul.Tensor','args':[{'ref':'activation'},{'ref':'up'}],'kwargs':{}},
               linear('reuse',gate,ub)]
        spec={'inputs':[{'name':'x','shape':[4,8]}],'nodes':nodes,'output':[{'ref':'product'},{'ref':'reuse'}]}
        with tempfile.TemporaryDirectory(prefix='cutline-dense-test-') as directory:
            path=Path(directory);(path/'graph.json').write_text(json.dumps(spec));mx.save_safetensors(str(path/'weights.safetensors'),weights)
            reference=MLXGraph(path,True,optimizations=())
            candidate=MLXGraph(path,True,optimizations=('dense','paired_dense'))
            self.assertIn(gate,candidate.weights)
            self.assertNotIn(up,candidate.weights)
            self.assertEqual(len(candidate.spec['nodes']),len(nodes)+1)
            shared=MLXGraph(path,True,shared_weights=candidate.weights,optimizations=('dense','paired_dense'))
            packed=gate.replace('_w_gate_weight','_paired_weight')
            self.assertIs(shared.weights[packed],candidate.weights[packed])
            for x in [mx.random.normal((4,8)),mx.random.normal((4,8))*2]:
                for a,b in zip(reference(x),candidate(x)):
                    np.testing.assert_allclose(np.array(a),np.array(b),atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
