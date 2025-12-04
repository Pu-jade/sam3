# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Mac/CPU compatible implementation of EDT.
Replaces Triton kernel with Scipy's distance_transform_edt.
"""

import torch
import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    raise ImportError("Running SAM3 on Mac requires scipy. Please install it: `pip install scipy`")

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    
    This version is modified for Mac (MPS/CPU) compatibility.
    It automatically moves data to CPU, computes EDT using Scipy, and moves it back.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.
              Non-zero elements are treated as the object (foreground).

    Returns:
        A tensor of the same shape as data containing the EDT.
        Equivalent to a batched version of cv2.distanceTransform.
    """
    # 1. 检查输入维度
    assert data.dim() == 3, f"Expected 3D tensor (B, H, W), got {data.shape}"
    
    # 2. 记录原始设备和类型 (比如 mps)
    original_device = data.device
    original_dtype = data.dtype

    # 3. 转移到 CPU 并转为 Numpy 布尔数组
    # Scipy EDT 计算的是“非零元素到最近零元素的距离”
    mask_np = data.detach().cpu().numpy() > 0 
    
    # 4. 准备输出数组
    # distance_transform_edt 默认返回 float64，我们转为 float32 节省显存
    output_np = np.zeros_like(mask_np, dtype=np.float32)

    # 5. 逐个批次计算
    # Scipy 默认把 3D 数组当做 3D 空间计算，但我们需要的是“一叠 2D 图片”
    # 所以需要循环处理 batch 维度
    B = mask_np.shape[0]
    for i in range(B):
        # 如果这一层全是 0 (背景)，结果就是全 0，跳过计算
        if not np.any(mask_np[i]):
            continue
        # 计算 L2 距离 (Euclidean)
        output_np[i] = distance_transform_edt(mask_np[i])

    # 6. 转回 Tensor 并放回原来的设备 (MPS)
    # 这一步会自动把数据从 CPU 显存复制回 Mac 的统一内存/显存
    return torch.from_numpy(output_np).to(device=original_device, dtype=torch.float32)