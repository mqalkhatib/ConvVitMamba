import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D, Dense, BatchNormalization,
    Activation, Add, Maximum, Concatenate, Average, AveragePooling2D, MaxPooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Reshape, InputLayer,
    MultiHeadAttention
)

def _safe_hwk(shape_like):
    """
    Return (H,W,C) robustly from a shape or list-of-shapes.
    - Accepts: tuple, tf.TensorShape, list, or list-of-shapes
    - Drops batch dim if present, converts None->1, pads to 3 dims.
    """
    # If it's a list-of-shapes, pick the first non-empty shape
    if isinstance(shape_like, (list, tuple)) and len(shape_like) > 0:
        # If first element looks like a shape (tuple/list/TensorShape), use it
        if isinstance(shape_like[0], (list, tuple, tf.TensorShape)):
            shape_like = shape_like[0]

    # Convert TensorShape to list
    if isinstance(shape_like, tf.TensorShape):
        shp = list(shape_like.as_list())
    else:
        try:
            shp = list(shape_like)
        except Exception:
            # fall back: unknown -> scalar
            shp = []

    # Drop batch if present
    if len(shp) > 0 and shp[0] is None:
        shp = shp[1:]

    # Convert None->1; ignore any nested tuples that may have slipped through
    norm = []
    for s in shp:
        if isinstance(s, (list, tuple, tf.TensorShape)):
            # if a nested shape appears, ignore and continue (rare)
            continue
        norm.append(1 if s is None else int(s))
    shp = norm

    # Pad/truncate to (H,W,C)
    if len(shp) == 0:
        shp = [1, 1, 1]
    elif len(shp) == 1:
        shp = [shp[0], 1, 1]
    elif len(shp) == 2:
        shp = [shp[0], shp[1], 1]
    else:
        shp = shp[:3]

    return tuple(shp)

def _prod(xs):
    p = 1
    for v in xs:
        p *= max(1, int(v))
    return p

def net_flops(model, table=False):
    if table:
        print('%25s | %18s | %18s | %14s | %12s | %10s | %12s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel', 'Filters', 'Strides', 'FLOPs'))
        print('-' * 130)

    total_flops = 0.0
    total_macc  = 0.0

    for l in model.layers:
        lname = l.name
        ltype = l.__class__.__name__
        flops = 0.0
        macc  = 0.0

        # Try to get shapes robustly
        try:
            ishape = getattr(l, "input_shape", None)
            oshape = getattr(l, "output_shape", None)
        except:
            ishape = None
            oshape = None

        # Handle list inputs (e.g., Add/Concat)
        if isinstance(ishape, list) and len(ishape) > 0:
            ish0 = ishape[0]
        else:
            ish0 = ishape

        H, W, C = _safe_hwk(ish0)
        Ho, Wo, Co = _safe_hwk(oshape)

        # --- Layer-specific rules ---

        if isinstance(l, InputLayer):
            flops = 0

        elif isinstance(l, Reshape):
            flops = 0

        elif isinstance(l, (Add, Maximum)):
            # element-wise operations: cost ~ number of elements in output * (num_inputs-1)
            n_in = len(ishape) if isinstance(ishape, list) else 1
            elems = _prod((Ho, Wo, Co))
            flops = (max(1, n_in) - 1) * elems

        elif isinstance(l, Average) and "pool" not in ltype.lower():
            n_in = len(ishape) if isinstance(ishape, list) else 1
            elems = _prod((Ho, Wo, Co))
            flops = max(1, n_in) * elems

        elif isinstance(l, Concatenate):
            # just data movement; ignore or count as adds across channels
            flops = 0

        elif isinstance(l, BatchNormalization):
            # scale + shift per element (2 ops). Some count 4 (mean/var), but at inference mean/var are constants.
            elems = _prod((Ho, Wo, Co))
            flops = 2 * elems

        elif isinstance(l, Activation):
            # optional: count 1 op per element
            elems = _prod((Ho, Wo, Co))
            flops = elems

        elif isinstance(l, (AveragePooling2D, MaxPooling2D)):
            # compare/adds within window per output element ~ kH*kW*C
            kH, kW = (l.pool_size if hasattr(l, "pool_size") else (1,1))
            elems_out = _prod((Ho, Wo, Co))
            flops = kH * kW * Co * Ho * Wo

        elif isinstance(l, GlobalAveragePooling2D) or isinstance(l, GlobalMaxPooling2D):
            # sum or compare across H*W per channel
            elems = H * W * C
            flops = elems

        elif isinstance(l, Flatten):
            flops = 0

        elif isinstance(l, Dense):
            units = l.units
            # infer input features
            if isinstance(ishape, (list, tuple)):
                in_feat = ishape[-1]
            else:
                # fall back to product of last dims if rank>2
                if isinstance(ish0, (list, tuple)) and len(ish0) > 1:
                    in_feat = ish0[-1]
                else:
                    in_feat = C
            in_feat = 1 if in_feat is None else int(in_feat)
            flops = 2.0 * in_feat * units
            macc  = flops / 2.0

        elif isinstance(l, Conv2D) and not isinstance(l, (DepthwiseConv2D, SeparableConv2D)):
            kH, kW = l.kernel_size
            sH, sW = l.strides
            Cin = C
            Cout = l.filters if l.filters is not None else Cin
            # conv FLOPs: 2 * (kH*kW*Cin) * (Ho*Wo*Cout)
            flops = 2.0 * (kH * kW * Cin) * (Ho * Wo * Cout)
            macc  = flops / 2.0

        elif isinstance(l, DepthwiseConv2D):
            kH, kW = l.kernel_size
            # depthwise: each input channel has its own kH*kW kernel -> Ho*Wo*Cin*(kH*kW) MACs
            flops = 2.0 * (kH * kW) * (Ho * Wo * C)  # C here equals Cin and Cout depthwise
            macc  = flops / 2.0

        elif isinstance(l, SeparableConv2D):
            # depthwise part + pointwise 1x1
            kH, kW = l.kernel_size
            Cin = C
            Cout = l.filters if l.filters is not None else Cin
            depthwise = 2.0 * (kH * kW) * (Ho * Wo * Cin)
            pointwise = 2.0 * (1 * 1 * Cin) * (Ho * Wo * Cout)
            flops = depthwise + pointwise
            macc  = flops / 2.0

        elif isinstance(l, MultiHeadAttention):
            # Approximate FLOPs for self-attention with seq=N, dim=D, heads=h
            # Projections: Q,K,V and output: 4 * N * D * D
            # Attn scores and weighted sums per head: 2 * h * N*N * (D/h)
            # Total ~ 4ND^2 + 2N^2D
            # Shapes are (B, N, D)
            # Try to read token length N and D from input
            # If input is list(tensor_q, tensor_v) this becomes trickier; in our model it's self-attention.
            if isinstance(ishape, list):
                shp = ishape[0]
            else:
                shp = ishape
            if isinstance(shp, (list, tuple)) and len(shp) >= 3:
                N = (1 if shp[1] is None else int(shp[1]))
                D = (1 if shp[2] is None else int(shp[2]))
            else:
                # fallback from output
                if isinstance(oshape, (list, tuple)) and len(oshape) >= 3:
                    N = (1 if oshape[1] is None else int(oshape[1]))
                    D = (1 if oshape[2] is None else int(oshape[2]))
                else:
                    N, D = Ho*Wo, Co
            flops = 4.0 * N * D * D + 2.0 * (N * N * D)
            macc  = flops / 2.0

        else:
            # Layers with negligible or undefined cost for this estimator
            flops = 0.0

        total_flops += flops
        total_macc  += macc

        if table:
            ks = getattr(l, "kernel_size", (0,0))
            st = getattr(l, "strides", (1,1))
            filters = getattr(l, "filters", 0)
            print('%25s | %18s | %18s | %14s | %12s | %10s | %12.1f' % (
                lname, str(ishape), str(oshape), str(ks), str(filters), str(st), flops))

    print("\nTotal FLOPs:  %d" % int(total_flops))
    print("Total MACCs:  %d" % int(total_macc))
    return int(total_flops), int(total_macc)
