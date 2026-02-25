# temporal_smoother.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


def _to_float(x):
    # metadata 里可能是 torch tensor
    try:
        import torch
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
    points_xy: [M,2] normalized (x,y). y expected monotonic (usually increasing).
    returns x_query: [len(y_query)] with np.nan where out of range.
    """
    if points_xy is None or len(points_xy) < 2:
        return np.full_like(y_query, np.nan, dtype=np.float32)

    xs = points_xy[:, 0].astype(np.float32)
    ys = points_xy[:, 1].astype(np.float32)

    # sort by y just in case
    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]

    # unique ys to avoid np.interp issues if duplicate y
    # (keep last occurrence)
    _, uniq_idx = np.unique(ys, return_index=True)
    ys_u = ys[uniq_idx]
    xs_u = xs[uniq_idx]

    if len(ys_u) < 2:
        return np.full_like(y_query, np.nan, dtype=np.float32)

    y_min, y_max = ys_u[0], ys_u[-1]
    x_query = np.interp(y_query, ys_u, xs_u).astype(np.float32)

    # mask out of range
    out = (y_query < y_min) | (y_query > y_max)
    x_query[out] = np.nan
    return x_query


def _lane_cost(curr_points: np.ndarray, prev_points: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float]:
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
    输出与 curr_points 同样数量的点，只对 x 做 EMA：
        x = alpha*x_curr + (1-alpha)*x_prev(y_curr)
    若 prev 在该 y 无定义，则保持 curr。
    """
    out = curr_points.copy().astype(np.float32)
    yq = out[:, 1].astype(np.float32)
    xp = _interp_x_at_y(prev_points, yq)
    mask = ~np.isnan(xp)
    out[mask, 0] = alpha * out[mask, 0] + (1.0 - alpha) * xp[mask]
    return out


@dataclass
class _Track:
    lane: Any                 # lib.lane.Lane
    age: int = 0              # 未匹配帧数
    hits: int = 0             # 匹配次数（可选）


class TemporalLaneSmoother:
    """
    推理期时序稳定：
    - 先对当前帧 lanes 与上一帧 tracks 做一对一匹配
    - 匹配上的 lane 做 EMA 平滑（默认只平滑 x）
    - 未匹配 track 可以保留 max_age 帧用于短暂缺失（可选）
    """

    def __init__(
        self,
        alpha: float = 0.8,           # EMA 权重：越大越“跟当前帧”
        match_thr: float = 0.08,      # 匹配阈值（单位：归一化 x 差，0.08 ~ 0.12 常用）
        min_overlap: float = 0.35,    # y 覆盖率不足则不匹配
        max_age: int = 0,             # 允许 track 失配保留几帧（0 = 不保留）
        y_samples: int = 50,          # cost 采样密度
        smooth_conf: bool = True,     # 是否平滑 metadata['conf']
        conf_beta: float = 0.8,       # conf EMA：beta 越大越“跟当前帧”
    ):
        self.alpha = float(alpha)
        self.match_thr = float(match_thr)
        self.min_overlap = float(min_overlap)
        self.max_age = int(max_age)
        self.y_grid = np.linspace(0.0, 1.0, int(y_samples), dtype=np.float32)

        self.smooth_conf = bool(smooth_conf)
        self.conf_beta = float(conf_beta)

        self._tracks: List[_Track] = []

    def reset(self):
        self._tracks = []

    def update(self, lanes: List[Any]) -> List[Any]:
        """
        输入：当前帧 lanes（Lane 对象列表）
        输出：平滑后的 lanes（同类型 Lane 对象列表）
        """
        if lanes is None:
            lanes = []

        # 如果没有历史 track，直接初始化
        if len(self._tracks) == 0:
            self._tracks = [_Track(lane=l, age=0, hits=1) for l in lanes]
            return lanes

        # 计算代价矩阵
        prev_lanes = [t.lane for t in self._tracks]
        C = np.full((len(lanes), len(prev_lanes)), np.inf, dtype=np.float32)
        O = np.zeros((len(lanes), len(prev_lanes)), dtype=np.float32)

        for i, cur in enumerate(lanes):
            cp = np.asarray(cur.points, dtype=np.float32)
            for j, prv in enumerate(prev_lanes):
                pp = np.asarray(prv.points, dtype=np.float32)
                cost, ov = _lane_cost(cp, pp, self.y_grid)
                C[i, j] = cost
                O[i, j] = ov

        # greedy 一对一匹配（足够用了，后面你要 Hungarian 我也可以给）
        matched_cur = set()
        matched_prev = set()
        pairs: List[Tuple[int, int]] = []

        # 按 cost 从小到大挑
        flat = [(C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1])]
        flat.sort(key=lambda x: x[0])

        for cost, i, j in flat:
            if i in matched_cur or j in matched_prev:
                continue
            if not np.isfinite(cost):
                continue
            if cost > self.match_thr:
                break
            if O[i, j] < self.min_overlap:
                continue
            matched_cur.add(i)
            matched_prev.add(j)
            pairs.append((i, j))

        # 生成输出 lanes（平滑匹配到的）
        out_lanes: List[Any] = []
        new_tracks: List[_Track] = []

        # 先把匹配到的处理
        for i, j in pairs:
            cur = lanes[i]
            prv_track = self._tracks[j]
            prv = prv_track.lane

            # 平滑 points
            cp = np.asarray(cur.points, dtype=np.float32)
            pp = np.asarray(prv.points, dtype=np.float32)
            sp = _smooth_lane_points(cp, pp, alpha=self.alpha)

            # 写回（保留 Lane 类型）
            cur.points = sp

            # 平滑 conf（可选）
            if self.smooth_conf and hasattr(cur, "metadata") and hasattr(prv, "metadata"):
                if isinstance(cur.metadata, dict) and isinstance(prv.metadata, dict):
                    if "conf" in cur.metadata and "conf" in prv.metadata:
                        c_now = _to_float(cur.metadata["conf"])
                        c_pre = _to_float(prv.metadata["conf"])
                        if isinstance(c_now, (int, float)) and isinstance(c_pre, (int, float)):
                            c_sm = self.conf_beta * c_now + (1.0 - self.conf_beta) * c_pre
                            cur.metadata["conf"] = c_sm

            out_lanes.append(cur)
            new_tracks.append(_Track(lane=cur, age=0, hits=prv_track.hits + 1))

        # 未匹配到的当前 lanes：直接输出并变成新 track
        for i, cur in enumerate(lanes):
            if i in matched_cur:
                continue
            out_lanes.append(cur)
            new_tracks.append(_Track(lane=cur, age=0, hits=1))

        # 未匹配到的历史 tracks：可选保留 max_age 帧（用于短暂漏检）
        if self.max_age > 0:
            for j, tr in enumerate(self._tracks):
                if j in matched_prev:
                    continue
                tr.age += 1
                if tr.age <= self.max_age:
                    # 你可以选择把它也输出（用于“补线”），也可以只保留不输出
                    # 这里默认：不输出，只保留
                    new_tracks.append(tr)

        # 更新 tracks
        self._tracks = new_tracks

        return out_lanes
