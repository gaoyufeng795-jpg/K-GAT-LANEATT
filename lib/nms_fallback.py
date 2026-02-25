# import torch
# from torchvision.ops import nms as tv_nms
#
#
# def nms(proposals: torch.Tensor, scores: torch.Tensor, overlap: float = 0.5, top_k: int | None = 3000):
#     """
#     Fallback NMS implementation using torchvision.ops.nms over coarse boxes.
#     Matches the extension's call signature and return shape:
#       returns (keep_indices, num_to_keep, None)
#
#     - proposals: Tensor [N, 5 + S], where proposals[:, 5:] are xs along S anchor rows
#     - scores:    Tensor [N]
#     - overlap:   IoU threshold
#     - top_k:     keep at most top_k indices
#     """
#     if proposals.numel() == 0:
#         device = proposals.device
#         return torch.empty((0,), dtype=torch.long, device=device), 0, None
#
#     # Build simple axis-aligned boxes using x-range; y is [0, 1] for all boxes.
#     xs = proposals[:, 5:]
#     # If proposals were in double, cast to float for tv_nms
#     xs = xs.float()
#     xmin, _ = xs.min(dim=1)
#     xmax, _ = xs.max(dim=1)
#     # Ensure proper ordering
#     x1 = torch.minimum(xmin, xmax)
#     x2 = torch.maximum(xmin, xmax)
#     # Avoid degenerate boxes (x2 == x1) that can produce NaNs in IoU
#     eps = 1e-6
#     x2 = torch.where(x2 <= x1, x1 + eps, x2)
#
#     y1 = torch.zeros_like(x1)
#     y2 = torch.ones_like(x2)
#
#     boxes = torch.stack([x1, y1, x2, y2], dim=1)
#
#     keep = tv_nms(boxes, scores.float(), overlap)
#     if top_k is not None:
#         keep = keep[:top_k]
#     num_to_keep = int(keep.numel())
#     return keep, num_to_keep, None
#
