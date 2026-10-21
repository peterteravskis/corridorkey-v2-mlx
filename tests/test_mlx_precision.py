"""Precision boundaries that matter for reduced-precision graph execution."""
import unittest
import json
from pathlib import Path
import tempfile
import mlx.core as mx
import numpy as np
from corridorkey_mlx.mlx_graph import MLXGraph, execute_fp16, floating_cast

class PrecisionTests(unittest.TestCase):
    def test_q8_packed_linear_matches_dequantized_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mx.random.seed(17)
            w=mx.random.normal((64,128));x=mx.random.normal((2,3,128)).astype(mx.float16)
            packed,scales,biases=mx.quantize(w,group_size=64,bits=8)
            scales=scales.astype(mx.float16);biases=biases.astype(mx.float16)
            bias=mx.random.normal((64,)).astype(mx.float16)
            weights={'p_weight':packed,'p_weight__scales':scales,'p_weight__biases':biases,'p_bias':bias}
            mx.save_safetensors(str(root/'weights.safetensors'),weights)
            spec={'precision':'q8-fp16-v1','quantized_weights':{'p_weight':{'group_size':64,'bits':8,'shape':[64,128]}},
                  'inputs':[{'name':'x','shape':[2,3,128]}],
                  'nodes':[{'name':'out','op':'aten.linear.default','args':[{'ref':'x'},{'ref':'p_weight'},{'ref':'p_bias'}],'kwargs':{}}],
                  'output':[{'ref':'out'}]}
            (root/'graph.json').write_text(json.dumps(spec))
            expected=x.astype(mx.float32)@mx.dequantize(packed,scales,biases,group_size=64,bits=8).astype(mx.float32).T+bias
            graph=MLXGraph(root,compile_graph=True);result=graph(x)[0]
            self.assertEqual(result.dtype,mx.float32)
            np.testing.assert_allclose(np.array(result),np.array(expected),atol=.05,rtol=.01)
            shared=MLXGraph(root,shared_weights=graph.weights)
            np.testing.assert_allclose(np.array(shared(x)[0]),np.array(result),atol=.02,rtol=.002)
    def test_float_cast_preserves_indices_and_masks(self):
        values=[mx.array([1.2]),mx.array([100000],dtype=mx.int64),mx.array([True])]
        result=floating_cast(values,mx.float16)
        self.assertEqual([x.dtype for x in result],[mx.float16,mx.int64,mx.bool_])
        self.assertEqual(result[1].item(),100000)
    def test_mean_avoids_half_sum_overflow(self):
        x=mx.full((10000,),100.,dtype=mx.float16)
        y=execute_fp16('aten.mean.default',[x,[0],False],{})
        self.assertEqual(y.dtype,mx.float16)
        self.assertEqual(y.item(),100.)
    def test_norm_large_inputs_is_finite(self):
        x=mx.array([[300,301,302,303]],dtype=mx.float16)
        y=execute_fp16('aten.layer_norm.default',[x,[4],None,None,1e-5],{})
        self.assertTrue(bool(mx.all(mx.isfinite(y))))
        np.testing.assert_allclose(np.array(y),[[-1.34164,-.44721,.44721,1.34164]],atol=.001)
    def test_graph_precision_is_opt_in_and_outputs_remain_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            graph={'inputs':[{'name':'x','shape':[1,2]}],
                   'nodes':[{'name':'out','op':'aten.linear.default','args':[{'ref':'x'},{'ref':'p_weight'}],'kwargs':{}}],
                   'output':[{'ref':'out'}]}
            weight=mx.array([[1.234567,.765432]])
            x=mx.array([[.456789,1.345678]])
            mx.save_safetensors(str(root/'weights.safetensors'),{'p_weight':weight})
            (root/'graph.json').write_text(json.dumps(graph))
            baseline=MLXGraph(root)(x)[0]
            np.testing.assert_array_equal(np.array(baseline),np.array(x@weight.T))
            graph['precision']='fp16-mixed-v1'
            (root/'graph.json').write_text(json.dumps(graph))
            mx.save_safetensors(str(root/'weights.safetensors'),{'p_weight':weight.astype(mx.float16)})
            candidate=MLXGraph(root,compile_graph=True)(x)[0]
            self.assertEqual(candidate.dtype,mx.float32)
            expected=(x.astype(mx.float16)@weight.astype(mx.float16).T).astype(mx.float32)
            np.testing.assert_array_equal(np.array(candidate),np.array(expected))

if __name__=='__main__':unittest.main()
