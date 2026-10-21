"""Numerical and fallback contracts for custom FP32 inference kernels."""
import unittest
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from corridorkey_mlx.mlx_graph import execute, execute_optimized, DEFAULT_OPTIMIZATIONS
from corridorkey_mlx.mlx_kernels import reconstruct, reflect_pad, reflect_pad_layout, reconstruct_half_experimental, channel_conv7_pair


class MLXKernelTests(unittest.TestCase):
    def test_channel_conv7_two_outputs_bias_strides_and_edges(self):
        torch.manual_seed(719)
        for batch,channels,height,width in [(1,3,7,7),(2,5,11,17),(1,33,9,10)]:
            source=torch.randn(batch,channels,height,width)*.2
            weight=torch.randn(channels*2,1,7,7)*.1
            for bias in [None,torch.arange(channels*2,dtype=torch.float32)*.031]:
                expected=F.conv2d(source,weight,bias,groups=channels).numpy()
                values=[mx.array(source.numpy()),mx.array(source.permute(0,2,3,1).contiguous().numpy()).transpose(0,3,1,2)]
                for x in values:
                    w=mx.array(weight.numpy());b=None if bias is None else mx.array(bias.numpy())
                    actual=mx.compile(lambda x,w:channel_conv7_pair(x,w,b))(x,w)
                    np.testing.assert_allclose(np.array(actual),expected,atol=3e-6,rtol=2e-5)

    def test_channel_conv7_dispatch_falls_back_for_stride_and_precision(self):
        x=mx.ones((1,3,11,13));w=mx.ones((6,1,7,7))*.01;b=mx.arange(6,dtype=mx.float32)
        for dtype,stride in [(mx.float32,[2,2]),(mx.float16,[1,1])]:
            args=[x.astype(dtype),w.astype(dtype),b.astype(dtype),stride,[0,0],[1,1],3]
            np.testing.assert_array_equal(np.array(execute_optimized('aten.conv2d.default',args,{},('channel_conv7',))),
                                          np.array(execute('aten.conv2d.default',args,{})))

    def test_half_kernels_remain_opt_in_and_padding_is_exact(self):
        self.assertNotIn('half_kernels',DEFAULT_OPTIMIZATIONS)
        x=mx.arange(2*5*7*3,dtype=mx.float16).reshape(2,5,7,3).transpose(0,3,1,2)
        expected=mx.pad(x,[(0,0),(0,0),(1,3),(2,1)],mode='reflect')
        for kernel in [reflect_pad,reflect_pad_layout]:
            result=kernel(x,[2,1,1,3])
            self.assertEqual(result.dtype,mx.float16)
            np.testing.assert_array_equal(np.array(result),np.array(expected))

    def test_half_reconstruction_bias_is_applied_to_channels(self):
        x=mx.zeros((2,7,3,5),dtype=mx.float16)
        w=mx.ones((7,4,32,32),dtype=mx.float16)*.01
        b=mx.array([.1,.2,.3,.4],dtype=mx.float16)
        actual=reconstruct_half_experimental(x,w,b)
        expected=execute('aten.conv_transpose2d.input',[x,w,b,[16,16],[8,8]],{})
        np.testing.assert_array_equal(np.array(actual),np.array(expected))

    def test_reconstruction_edges_batches_bias_and_noncontiguous(self):
        torch.manual_seed(582)
        for batch, h, w, stride, pad in [(1,1,1,4,2),(2,3,5,4,2),(1,5,3,16,8)]:
            x=torch.randn(batch,7,h,w)
            weight=torch.randn(7,4,2*stride,2*stride)*.1
            for bias in [None,torch.randn(4)]:
                expected=F.conv_transpose2d(x,weight,bias,stride=stride,padding=pad).numpy()
                # NHWC backing with NCHW view, as used by graph feature tensors.
                a=mx.array(x.permute(0,2,3,1).contiguous().numpy()).transpose(0,3,1,2)
                b=mx.array(weight.numpy());c=None if bias is None else mx.array(bias.numpy())
                result=mx.compile(lambda a,b: reconstruct(a,b,c,stride,pad))(a,b)
                np.testing.assert_allclose(np.array(result),expected,atol=3e-6,rtol=2e-5)

    def test_padding_matches_torch_for_strides_and_boundaries(self):
        x=torch.randn(2,3,5,7)
        views=[mx.array(x.numpy()),mx.array(x.permute(0,2,3,1).contiguous().numpy()).transpose(0,3,1,2)]
        for pads in [[0,0,0,0],[2,1,1,3],[6,6,4,4]]:
            expected=F.pad(x,pads,mode='reflect').numpy()
            for view in views:
                for kernel in [reflect_pad,reflect_pad_layout]:
                    np.testing.assert_array_equal(np.array(mx.compile(lambda a:kernel(a,pads))(view)),expected)
        strided=mx.arange(2*3*10*14,dtype=mx.float32).reshape(2,3,10,14)[:,:,::2,::2]
        np.testing.assert_array_equal(np.array(reflect_pad_layout(strided,[2,1,1,3])),
            np.array(mx.pad(strided,[(0,0),(0,0),(1,3),(2,1)],mode='reflect')))

    def test_unsupported_convolution_and_precision_use_reference(self):
        x=mx.ones((1,3,2,4));w=mx.ones((3,4,8,8))*.01;b=mx.zeros((4,))
        args=[x,w,b,[4,4],[2,2]]
        opts=('reconstruction','padding','layout')
        np.testing.assert_array_equal(np.array(execute_optimized('aten.conv_transpose2d.input',args,{},opts)),
                                      np.array(execute('aten.conv_transpose2d.input',args,{})))
        x=x.astype(mx.float16)
        args=[x,[1,1,1,1],'reflect']
        np.testing.assert_array_equal(np.array(execute_optimized('aten.pad.default',args,{},opts)),
                                      np.array(execute('aten.pad.default',args,{})))


if __name__=='__main__':unittest.main()
