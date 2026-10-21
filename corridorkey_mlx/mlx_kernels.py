# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Equivalent FP32 execution kernels for the exported inference graphs."""
import mlx.core as mx


_overlap_add = mx.fast.metal_kernel(
    name="cutline_overlap_add", input_names=["patches", "bias"], output_names=["out"],
    source=r"""
    uint idx = thread_position_in_grid.x;
    if (idx >= B*C*OH*OW) return;
    int ox = idx % OW, oy = (idx / OW) % OH;
    int c = (idx / (OW*OH)) % C, b = idx / (OW*OH*C);
    int gy = oy + P, gx = ox + P;
    int by = gy / S, bx = gx / S;
    float value = 0;
    for (int dy = 1; dy >= 0; --dy) {
        int py = by-dy, ky = gy-py*S;
        if (py < 0 || py >= H) continue;
        for (int dx = 1; dx >= 0; --dx) {
            int px = bx-dx, kx = gx-px*S;
            if (px < 0 || px >= W) continue;
            value += patches[((((b*H+py)*W+px)*C+c)*K+ky)*K+kx];
        }
    }
    out[idx] = value + bias[c];
    """)


def reconstruct(x, weight, bias, stride=16, padding=8):
    """Project tokens into overlapping patches, then gather each output once.

    Only the 2*stride square kernel, groups=1 case is supported. Accumulation
    stays FP32; no atomics, approximate weights or reduced spatial context.
    """
    batch, channels, height, width = x.shape
    _, out_channels, kh, kw = weight.shape
    assert x.dtype == weight.dtype == mx.float32
    assert kh == kw == 2*stride
    patches = x.transpose(0, 2, 3, 1) @ weight.reshape(channels, -1)
    oh = (height-1)*stride+kh-2*padding
    ow = (width-1)*stride+kw-2*padding
    if bias is None:
        bias = mx.zeros((out_channels,), dtype=mx.float32)
    return _overlap_add(
        inputs=[patches, bias],
        template=[('B', batch), ('C', out_channels), ('H', height), ('W', width),
                  ('K', kh), ('S', stride), ('P', padding), ('OH', oh), ('OW', ow)],
        grid=(batch*out_channels*oh*ow, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(batch, out_channels, oh, ow)], output_dtypes=[mx.float32])[0]


_reflect = mx.fast.metal_kernel(
    name="cutline_reflect_pad", input_names=["inp"], output_names=["out"],
    source=r"""
    uint idx = thread_position_in_grid.x;
    if (idx >= B*C*OH*OW) return;
    int ox = idx % OW, oy = (idx / OW) % OH;
    int bc = idx / (OW*OH);
    int y = oy-TOP, x = ox-LEFT;
    y = y < 0 ? -y : (y >= H ? 2*H-2-y : y);
    x = x < 0 ? -x : (x >= W ? 2*W-2-x : x);
    out[idx] = inp[(bc*H+y)*W+x];
    """)


def reflect_pad(x, pads):
    """One gather for spatial NCHW reflect padding; matches PyTorch bounds."""
    left, right, top, bottom = pads
    batch, channels, height, width = x.shape
    assert x.dtype in (mx.float32, mx.float16)
    assert 0 <= left < width and 0 <= right < width
    assert 0 <= top < height and 0 <= bottom < height
    oh, ow = height+top+bottom, width+left+right
    return _reflect(
        inputs=[x], template=[('B', batch), ('C', channels), ('H', height),
                             ('W', width), ('OH', oh), ('OW', ow),
                             ('TOP', top), ('LEFT', left)],
        grid=(batch*channels*oh*ow, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(batch, channels, oh, ow)], output_dtypes=[x.dtype])[0]


_reflect_layout = mx.fast.metal_kernel(
    name="cutline_reflect_pad_channels_last", input_names=["inp"], output_names=["out"],
    ensure_row_contiguous=False,
    source=r"""
    uint idx = thread_position_in_grid.x;
    if (idx >= B*C*OH*OW) return;
    int c = idx % C, ox = (idx/C) % OW;
    int oy = (idx/(C*OW)) % OH, b = idx/(C*OW*OH);
    int y = oy-TOP, x = ox-LEFT;
    y = y < 0 ? -y : (y >= H ? 2*H-2-y : y);
    x = x < 0 ? -x : (x >= W ? 2*W-2-x : x);
    out[idx] = inp[b*inp_strides[0]+c*inp_strides[1]+y*inp_strides[2]+x*inp_strides[3]];
    """)


def reflect_pad_layout(x, pads):
    """Pad directly into MLX convolution layout, retaining an NCHW view contract.

    The stride-aware read accepts upstream NHWC-backed views without copying
    them to NCHW first. The following conv's NCHW-to-NHWC transpose is a view.
    """
    left, right, top, bottom = pads
    batch, channels, height, width = x.shape
    assert x.dtype in (mx.float32, mx.float16)
    assert 0 <= left < width and 0 <= right < width
    assert 0 <= top < height and 0 <= bottom < height
    oh, ow = height+top+bottom, width+left+right
    out = _reflect_layout(
        inputs=[x], template=[('B', batch), ('C', channels), ('H', height),
                             ('W', width), ('OH', oh), ('OW', ow),
                             ('TOP', top), ('LEFT', left)],
        grid=(batch*channels*oh*ow, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(batch, oh, ow, channels)], output_dtypes=[x.dtype])[0]
    return out.transpose(0, 3, 1, 2)


def reconstruct_half_experimental(x, weight, bias):
    """Opt-in half-weight projection with FP32 intermediate accumulation.

    Round convolution output before the half-precision bias addition, matching
    the existing operator boundaries. This is not enabled in production defaults.
    """
    out = reconstruct(x.astype(mx.float32), weight.astype(mx.float32), None).astype(mx.float16)
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


_channel_conv7 = mx.fast.metal_kernel(
    name='cutline_channel_conv7_pair',input_names=['inp','weight','bias'],output_names=['out'],
    source=r'''
    uint idx=thread_position_in_grid.x;
    if (idx>=B*OH*OW*C) return;
    uint c=idx%C, px=(idx/C)%OW, py=(idx/(C*OW))%OH;
    uint b=idx/(C*OW*OH);
    float v0=0.0f, v1=0.0f;
    #pragma unroll
    for (int ky=0;ky<7;++ky) {
        #pragma unroll
        for (int kx=0;kx<7;++kx) {
            float x=inp[((b*IH+py+ky)*IW+px+kx)*C+c];
            uint wi=((ky*7+kx)*C+c)*2;
            v0=fma(x,float(weight[wi]),v0);
            v1=fma(x,float(weight[wi+1]),v1);
        }
    }
    out[idx*2]=T(T(v0)+bias[c*2]);
    out[idx*2+1]=T(T(v1)+bias[c*2+1]);
    ''')


def channel_conv7_pair(x, weight, bias):
    """Valid 7x7 depthwise convolution, two outputs per input channel.

    Accumulate both filters in one pass over the input. The output has NHWC
    backing with the existing NCHW view contract. Reflection padding is already
    present in x. Native convolution remains the fallback for all other shapes.
    FP16 rounds the FP32 accumulation before adding bias, preserving the native
    convolution/bias boundary. Its dispatch is separate from the older,
    experimental half-precision reconstruction kernels.
    """
    batch,c,ih,iw=x.shape
    assert x.dtype==weight.dtype and x.dtype in (mx.float32,mx.float16)
    assert weight.shape==(2*c,1,7,7) and ih>=7 and iw>=7
    oh,ow=ih-6,iw-6
    if bias is None:bias=mx.zeros((2*c,),dtype=x.dtype)
    packed=weight.reshape(2*c,49).T
    out=_channel_conv7(inputs=[x.transpose(0,2,3,1),packed,bias],
        template=[('T',x.dtype),('B',batch),('C',c),('IH',ih),('IW',iw),('OH',oh),('OW',ow)],
        grid=(batch*oh*ow*c,1,1),threadgroup=(256,1,1),
        output_shapes=[(batch,oh,ow,2*c)],output_dtypes=[x.dtype])[0]
    return out.transpose(0,3,1,2)
