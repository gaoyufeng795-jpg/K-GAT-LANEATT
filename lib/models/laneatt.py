import math

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34

# Try native CUDA/C++ NMS first; fall back to a pure-PyTorch implementation if it fails
try:
    from lib.nms.src.nms import nms as nms_impl  # compiled extension
except Exception:
    from lib.nms_fallback import nms as nms_impl  # python fallback using torchvision.ops

from lib.lane import Lane
from lib.focal_loss import FocalLoss

from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets


# ============================================================
# Temporal smoother (A): greedy matching + EMA on x(y)
# Self-contained, used only at inference (decode-as-lanes)
# ============================================================

def _to_float(x):
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return x


def _interp_x_at_y(points_xy: np.ndarray, y_query: np.ndarray) -> np.ndarray:
    """
    points_xy: [M,2] normalized (x,y)
    returns: x_query with NaN where y out-of-range
    """
    if points_xy is None or len(points_xy) < 2:
        return np.full_like(y_query, np.nan, dtype=np.float32)

    xs = points_xy[:, 0].astype(np.float32)
    ys = points_xy[:, 1].astype(np.float32)

    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]

    # remove duplicate ys (keep first occurrence)
    _, uniq_idx = np.unique(ys, return_index=True)
    ys_u = ys[uniq_idx]
    xs_u = xs[uniq_idx]

    if len(ys_u) < 2:
        return np.full_like(y_query, np.nan, dtype=np.float32)

    y_min, y_max = ys_u[0], ys_u[-1]
    x_query = np.interp(y_query, ys_u, xs_u).astype(np.float32)
    out = (y_query < y_min) | (y_query > y_max)
    x_query[out] = np.nan
    return x_query


def _lane_cost(curr_points: np.ndarray, prev_points: np.ndarray, y_grid: np.ndarray):
    """
    cost: mean |dx| on overlapping y positions
    overlap: fraction of y_grid where both valid
    """
    xc = _interp_x_at_y(curr_points, y_grid)
    xp = _interp_x_at_y(prev_points, y_grid)
    valid = (~np.isnan(xc)) & (~np.isnan(xp))
    overlap = float(valid.mean()) if valid.size > 0 else 0.0
    if valid.sum() == 0:
        return float("inf"), overlap
    cost = float(np.mean(np.abs(xc[valid] - xp[valid])))
    return cost, overlap


def _smooth_lane_points(curr_points: np.ndarray, prev_points: np.ndarray, alpha: float) -> np.ndarray:
    """
    Keep the same number of points as curr_points.
    EMA only on x at the curr y's: x = alpha*x_curr + (1-alpha)*x_prev(y_curr)
    """
    out = curr_points.copy().astype(np.float32)
    yq = out[:, 1].astype(np.float32)
    xp = _interp_x_at_y(prev_points, yq)
    mask = ~np.isnan(xp)
    out[mask, 0] = alpha * out[mask, 0] + (1.0 - alpha) * xp[mask]
    return out


class TemporalLaneSmoother:
    """
    Inference-only temporal smoother:
    - Greedy one-to-one matching with previous frame lanes
    - EMA smoothing on x(y)
    """

    def __init__(
        self,
        alpha: float = 0.85,
        match_thr: float = 0.10,
        min_overlap: float = 0.35,
        max_age: int = 0,          # keep unmatched history (not output), for short misses
        y_samples: int = 60,
        smooth_conf: bool = True,
        conf_beta: float = 0.8,
    ):
        self.alpha = float(alpha)
        self.match_thr = float(match_thr)
        self.min_overlap = float(min_overlap)
        self.max_age = int(max_age)
        self.y_grid = np.linspace(0.0, 1.0, int(y_samples), dtype=np.float32)
        self.smooth_conf = bool(smooth_conf)
        self.conf_beta = float(conf_beta)

        # tracks: list of dict {lane, age, hits}
        self._tracks = []

    def reset(self):
        self._tracks = []

    def update(self, lanes):
        if lanes is None:
            lanes = []

        if len(self._tracks) == 0:
            self._tracks = [{"lane": l, "age": 0, "hits": 1} for l in lanes]
            return lanes

        prev_lanes = [t["lane"] for t in self._tracks]
        M = len(lanes)
        P = len(prev_lanes)

        if M == 0:
            # age history
            if self.max_age > 0:
                new_tracks = []
                for t in self._tracks:
                    t["age"] += 1
                    if t["age"] <= self.max_age:
                        new_tracks.append(t)
                self._tracks = new_tracks
            else:
                self._tracks = []
            return []

        C = np.full((M, P), np.inf, dtype=np.float32)
        O = np.zeros((M, P), dtype=np.float32)

        for i, cur in enumerate(lanes):
            cp = np.asarray(cur.points, dtype=np.float32)
            for j, prv in enumerate(prev_lanes):
                pp = np.asarray(prv.points, dtype=np.float32)
                cost, ov = _lane_cost(cp, pp, self.y_grid)
                C[i, j] = cost
                O[i, j] = ov

        # greedy matching by ascending cost
        pairs = []
        used_i = set()
        used_j = set()
        flat = [(float(C[i, j]), i, j) for i in range(M) for j in range(P)]
        flat.sort(key=lambda x: x[0])

        for cost, i, j in flat:
            if i in used_i or j in used_j:
                continue
            if not np.isfinite(cost):
                continue
            if cost > self.match_thr:
                break
            if O[i, j] < self.min_overlap:
                continue
            used_i.add(i)
            used_j.add(j)
            pairs.append((i, j))

        out_lanes = []
        new_tracks = []

        # matched lanes: smooth and update track
        for i, j in pairs:
            cur = lanes[i]
            prv = self._tracks[j]["lane"]

            cp = np.asarray(cur.points, dtype=np.float32)
            pp = np.asarray(prv.points, dtype=np.float32)
            sp = _smooth_lane_points(cp, pp, alpha=self.alpha)
            cur.points = sp

            # smooth conf (optional)
            if self.smooth_conf and hasattr(cur, "metadata") and hasattr(prv, "metadata"):
                if isinstance(cur.metadata, dict) and isinstance(prv.metadata, dict):
                    if "conf" in cur.metadata and "conf" in prv.metadata:
                        c_now = _to_float(cur.metadata["conf"])
                        c_pre = _to_float(prv.metadata["conf"])
                        if isinstance(c_now, (int, float)) and isinstance(c_pre, (int, float)):
                            cur.metadata["conf"] = self.conf_beta * c_now + (1.0 - self.conf_beta) * c_pre

            out_lanes.append(cur)
            new_tracks.append({"lane": cur, "age": 0, "hits": self._tracks[j]["hits"] + 1})

        # unmatched current lanes: keep as-is
        for i, cur in enumerate(lanes):
            if i in used_i:
                continue
            out_lanes.append(cur)
            new_tracks.append({"lane": cur, "age": 0, "hits": 1})

        # unmatched previous tracks: optionally keep for a few frames (do not output by default)
        if self.max_age > 0:
            for j, t in enumerate(self._tracks):
                if j in used_j:
                    continue
                t["age"] += 1
                if t["age"] <= self.max_age:
                    new_tracks.append(t)

        self._tracks = new_tracks
        return out_lanes


# ============================================================
# LaneATT
# ============================================================

class LaneATT(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64,
                 # --- 新增：注意力相关超参（向后兼容） ---
                 attn_type: str = 'graph',       # 'mlp'(原实现) | 'graph'(邻域多头)
                 k_neighbors: int = 16,          # 每个锚只看K个几何邻居
                 num_heads: int = 4,             # 多头数
                 d_model: int = None,            # 注意力内部维度(默认=局部特征维)
                 geo_dim: int = 16,              # 几何嵌入维度

                 # --- 新增：Temporal smoother（推理期后处理） ---
                 temporal_smooth: bool = True,
                 temporal_alpha: float = 0.85,
                 temporal_match_thr: float = 0.10,
                 temporal_min_overlap: float = 0.35,
                 temporal_max_age: int = 1,
                 temporal_y_samples: int = 60,
                 ):
        super(LaneATT, self).__init__()
        # Some definitions
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors（同时记录每个锚的角度与边界ID用于几何先验）
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

            # 同步裁剪 buffer（避免破坏 buffer 属性）
            self._replace_buffer('anchor_meta', self.anchor_meta[ind].clone())
            self._replace_buffer('anchor_boundary_id', self.anchor_boundary_id[ind].clone())
            # 邻域索引稍后会基于裁剪后的meta重建

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)

        # 分类/回归头保持不变：输入= [全局特征, 局部特征] 拼接
        D_in = self.anchor_feat_channels * self.fmap_h
        self.cls_layer = nn.Linear(2 * D_in, 2)
        self.reg_layer = nn.Linear(2 * D_in, self.n_offsets + 1)

        # --- 注意力模块 ---
        self.attn_type = attn_type.lower()
        self.k_neighbors = int(k_neighbors)
        self.num_heads = int(num_heads)
        self.attn_dim = D_in if d_model is None else int(d_model)
        assert self.attn_dim % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.attn_dim // self.num_heads
        self.geo_dim = int(geo_dim)

        if self.attn_type == 'mlp':
            # 兼容原实现的“单边”MLP注意力（对照论文式(2)-(4)）
            self.attention_layer = nn.Linear(D_in, len(self.anchors) - 1)
            self.initialize_layer(self.attention_layer)
        else:
            # 邻域多头注意力（Anchor Graph Attention）
            self.q_proj = nn.Linear(D_in, self.attn_dim)
            self.k_proj = nn.Linear(D_in + self.geo_dim, self.attn_dim)
            self.v_proj = nn.Linear(D_in, self.attn_dim)
            self.out_proj = nn.Linear(self.attn_dim, D_in)
            # 几何嵌入 + 偏置：输入(Δx, Δy, Δsinθ, Δcosθ)
            self.geo_mlp = nn.Sequential(
                nn.Linear(4, 32), nn.ReLU(inplace=True),
                nn.Linear(32, self.geo_dim)
            )
            self.geo_bias = nn.Sequential(
                nn.Linear(4, self.num_heads)
            )
            self._build_neighbors()  # 基于当前 anchors 的 meta 构建 K 近邻索引

        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

        # --------------------------------------------------------
        # Temporal smoother: inference-only
        # --------------------------------------------------------
        self.temporal_smooth = bool(temporal_smooth)
        self._temporal_cfg = dict(
            alpha=float(temporal_alpha),
            match_thr=float(temporal_match_thr),
            min_overlap=float(temporal_min_overlap),
            max_age=int(temporal_max_age),
            y_samples=int(temporal_y_samples),
        )
        # one smoother per image in batch at decode-time
        self._temporal_smoothers = None  # type: list[TemporalLaneSmoother] | None

    # --------------------------------------------------------
    # buffer helper (safe replace)
    # --------------------------------------------------------
    def _replace_buffer(self, name: str, tensor: torch.Tensor):
        if not hasattr(self, "_buffers"):
            self.register_buffer(name, tensor)
            return
        if name in self._buffers:
            # remove old buffer then re-register
            del self._buffers[name]
        self.register_buffer(name, tensor)

    # --------------------------------------------------------
    # Temporal API
    # --------------------------------------------------------
    def reset_temporal(self):
        """Call this at the start of a new video/sequence."""
        if self._temporal_smoothers is not None:
            for s in self._temporal_smoothers:
                s.reset()

    def _ensure_temporal_smoothers(self, batch_size: int):
        if not self.temporal_smooth:
            return
        if (self._temporal_smoothers is None) or (len(self._temporal_smoothers) != batch_size):
            self._temporal_smoothers = [
                TemporalLaneSmoother(**self._temporal_cfg) for _ in range(batch_size)
            ]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        batch_anchor_features = self.cut_anchor_features(batch_features)

        # Join proposals from all images into a single proposals features batch
        D_in = self.anchor_feat_channels * self.fmap_h
        B = x.shape[0]
        N = len(self.anchors)
        batch_anchor_features = batch_anchor_features.view(-1, D_in)  # [B*N, D]

        if self.attn_type == 'mlp':
            # ====== 原始 MLP 注意力（LaneATT 论文式(2)-(4)）======
            softmax = nn.Softmax(dim=1)
            scores = self.attention_layer(batch_anchor_features)  # [B*N, N-1]
            attention = softmax(scores).reshape(B, N, -1)
            attention_matrix = torch.eye(N, device=x.device).repeat(B, 1, 1)
            non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
            attention_matrix[:] = 0
            attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
            batch_anchor_features_3d = batch_anchor_features.view(B, N, D_in)
            attention_features = torch.bmm(
                torch.transpose(batch_anchor_features_3d, 1, 2),
                torch.transpose(attention_matrix, 1, 2)
            ).transpose(1, 2)  # [B, N, D]
            attention_features = attention_features.reshape(-1, D_in)
            batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)
            batch_attention_matrix = attention_matrix  # 传给 nms（下游未使用，仅占位）
        else:
            # ====== 邻域多头注意力（轻量 Transformer / AGA）======
            feat = batch_anchor_features.view(B, N, D_in)  # [B,N,D]
            K = self.nbr_idx.size(1)
            feat_nbr = torch.stack([feat[b][self.nbr_idx] for b in range(B)], dim=0)  # [B,N,K,D]

            # 几何编码与偏置（常量 w.r.t. batch）
            meta = self.anchor_meta.to(feat.device)          # [N,4]  -> (x,y,sinθ,cosθ)
            meta_j = meta[self.nbr_idx]                      # [N,K,4]
            meta_i = meta.unsqueeze(1).expand(N, K, 4)       # [N,K,4]
            dx = (meta_i[..., 0] - meta_j[..., 0]).abs()
            dy = (meta_i[..., 1] - meta_j[..., 1]).abs()
            dsin = (meta_i[..., 2] - meta_j[..., 2]).abs()
            dcos = (meta_i[..., 3] - meta_j[..., 3]).abs()
            geo_ij = torch.stack([dx, dy, dsin, dcos], dim=-1)    # [N,K,4]
            geo_emb = self.geo_mlp(geo_ij).to(feat.device)        # [N,K,geo_dim]
            geo_b = self.geo_bias(geo_ij).to(feat.device)         # [N,K,H]

            # 多头投影
            Q = self.q_proj(feat).view(B, N, self.num_heads, self.head_dim)                # [B,N,H,Dh]
            Kp = self.k_proj(torch.cat([feat_nbr, geo_emb.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1))
            Kp = Kp.view(B, N, K, self.num_heads, self.head_dim)                           # [B,N,K,H,Dh]
            V = self.v_proj(feat_nbr).view(B, N, K, self.num_heads, self.head_dim)         # [B,N,K,H,Dh]

            # 注意力打分（只在邻域K上 softmax），加入几何偏置
            logits = (Q.unsqueeze(2) * Kp).sum(-1) / math.sqrt(self.head_dim)              # [B,N,K,H]
            logits = logits + geo_b.unsqueeze(0)                                           # [B,N,K,H]
            attn = torch.softmax(logits, dim=2)                                            # [B,N,K,H]

            # 聚合
            Aglob = (attn.unsqueeze(-1) * V).sum(dim=2).reshape(B, N, self.attn_dim)       # [B,N,d_model]
            Aglob = self.out_proj(Aglob)                                                   # [B,N,D]
            attention_features = Aglob.reshape(-1, D_in)                                    # [B*N,D]
            batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)
            batch_attention_matrix = torch.eye(N, device=x.device).repeat(B, 1, 1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(B, -1, cls_logits.shape[1])
        reg = reg.reshape(B, -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold)

        return proposals_list

    # --------------------------------------------------------
    # NMS / Loss
    # --------------------------------------------------------
    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms_impl(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                device = positives.device
                all_indices = torch.arange(num_positives, dtype=torch.long, device=device)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int, device=device)
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        cls_loss /= valid_imgs
        reg_loss /= valid_imgs
        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    # --------------------------------------------------------
    # Anchor-based pooling / anchors generation
    # --------------------------------------------------------
    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        n_proposals = len(self.anchors_cut)

        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut, left_theta, left_b = self.generate_side_anchors(
            self.left_angles, x=0., nb_origins=lateral_n, boundary_id=0)
        right_anchors, right_cut, right_theta, right_b = self.generate_side_anchors(
            self.right_angles, x=1., nb_origins=lateral_n, boundary_id=2)
        bottom_anchors, bottom_cut, bottom_theta, bottom_b = self.generate_side_anchors(
            self.bottom_angles, y=1., nb_origins=bottom_n, boundary_id=1)

        anchors = torch.cat([left_anchors, bottom_anchors, right_anchors])
        anchors_cut = torch.cat([left_cut, bottom_cut, right_cut])

        theta = torch.cat([left_theta, bottom_theta, right_theta])              # [N]
        b_id = torch.cat([left_b, bottom_b, right_b])                           # [N]
        x_start = anchors[:, 3].clone()
        y_start = anchors[:, 2].clone()
        meta = torch.stack([x_start, y_start, torch.sin(theta), torch.cos(theta)], dim=1)  # [N,4]

        self.register_buffer('anchor_meta', meta)
        self.register_buffer('anchor_boundary_id', b_id)

        return anchors, anchors_cut

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None, boundary_id=0):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        thetas = torch.zeros((n_anchors,))
        b_ids = torch.full((n_anchors,), int(boundary_id), dtype=torch.long)

        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)
                thetas[k] = angle * math.pi / 180.0

        return anchors, anchors_cut, thetas, b_ids

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w
        return anchor

    # --------------------------------------------------------
    # Utils
    # --------------------------------------------------------
    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)
        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)

            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        """
        If temporal_smooth=True and model.eval(), then decode(as_lanes=True)
        will additionally smooth lanes across calls (video frames).
        IMPORTANT: call model.reset_temporal() at the start of each new video/sequence.
        """
        softmax = nn.Softmax(dim=1)
        decoded = []

        # proposals_list is list of length B (one per image)
        if self.temporal_smooth and (not self.training) and as_lanes:
            self._ensure_temporal_smoothers(batch_size=len(proposals_list))

        for img_idx, (proposals, _, _, _) in enumerate(proposals_list):
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                pred = []
            else:
                pred = self.proposals_to_pred(proposals) if as_lanes else proposals

            # temporal smoothing only for lanes and only in eval mode
            if as_lanes and self.temporal_smooth and (not self.training) and self._temporal_smoothers is not None:
                pred = self._temporal_smoothers[img_idx].update(pred)

            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self

    # --------------------------------------------------------
    # 邻域构建（一次性，随 anchors 变化而重建）
    # --------------------------------------------------------
    def _build_neighbors(self):
        """
        基于 anchor_meta 构建按几何相似度排序的 Top-K 邻居索引（不含自身）。
        代价函数： |Δx| + |Δy| + λ * (1 - cosΔθ)
        """
        meta = self.anchor_meta  # [N,4] => (x,y,sinθ,cosθ)
        N = meta.size(0)
        x = meta[:, 0:1]
        y = meta[:, 1:2]
        sin_t = meta[:, 2:3]
        cos_t = meta[:, 3:4]

        cos_d = sin_t @ sin_t.t() + cos_t @ cos_t.t()              # [N,N]
        ang_cost = 1.0 - torch.clamp(cos_d, -1.0, 1.0)

        dx = (x - x.t()).abs()
        dy = (y - y.t()).abs()

        dist = dx + dy + 0.5 * ang_cost
        dist = dist + torch.eye(N, device=dist.device) * 1e6

        K = min(self.k_neighbors, N - 1)
        nbr_idx = torch.topk(dist, k=K, dim=1, largest=False).indices  # [N,K]
        self._replace_buffer('nbr_idx', nbr_idx)


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    elif backbone == 'resnet34':
        backbone = torch.nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride
