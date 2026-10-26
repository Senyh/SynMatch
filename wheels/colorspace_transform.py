import torch


def rgb_to_ycbcr_batch(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of RGB images to YCbCr color space (BT.601 standard).
    
    Args:
        image: Tensor of shape [B, 3, H, W], values in [0, 1]
        
    Returns:
        Tensor of shape [B, 3, H, W] in YCbCr space (also in [0, 1] range)
    """
    if image.ndim != 4 or image.size(1) != 3:
        raise ValueError("Input must be a [B, 3, H, W] RGB tensor.")
    
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb_batch(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of YCbCr images back to RGB.
    
    Args:
        ycbcr: Tensor of shape [B, 3, H, W], values in [0, 1]
    
    Returns:
        Tensor of shape [B, 3, H, W] in RGB space
    """
    y  = ycbcr[:, 0:1]
    cb = ycbcr[:, 1:2] - 0.5
    cr = ycbcr[:, 2:3] - 0.5

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)