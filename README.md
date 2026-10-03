# K-GAT-LaneATT

**Geometry-Aware Sparse Graph Attention for Lane Detection**

K-GAT-LaneATT extends [LaneATT](https://github.com/lucastabelini/LaneATT) by replacing its dense global anchor interaction with a **geometry-aware sparse graph attention head**.

Lane anchors are represented using their spatial position and orientation. We construct a static **k-nearest-neighbor (kNN) graph** in this geometric space and restrict attention to local, geometrically relevant anchor neighborhoods. This introduces an explicit lane-geometry prior while reducing anchor interaction complexity from \(O(N^2D)\) to \(O(NKD)\).

## Method

- Geometry-aware anchor representation using position and orientation
- Static **KNN graph with K = 16**
- Multi-head graph attention over local anchor neighborhoods
- Relative geometric features incorporated into anchor interaction
- Backbone, anchor definitions, training pipeline, losses, and NMS kept unchanged from LaneATT for controlled comparison

## Results

| Dataset | Backbone | Metric | LaneATT | K-GAT |
|---|---|---:|---:|---:|
| TuSimple | ResNet-18 | Accuracy | 95.57% | **95.76%** |
| TuSimple | ResNet-34 | Accuracy | 95.63% | **95.96%** |
| CULane | ResNet-18 | F1 | 75.07% | **75.90%** |
| LLAMAS | ResNet-18 | F1 | 93.46% | **94.66%** |
| LLAMAS | ResNet-34 | F1 | 93.74% | **95.02%** |

On TuSimple with ResNet-34, the measured validation runtime on an NVIDIA RTX 4070 decreased from **42.5 s to 38.1 s**, corresponding to a **10.3% reduction**.

## Installation and Usage

The project is built upon the original LaneATT codebase. Dataset preparation, training, and evaluation follow the original LaneATT pipeline.

Please refer to the [LaneATT repository](https://github.com/lucastabelini/LaneATT) for detailed setup instructions.

## Acknowledgements

This project builds upon the excellent work of Tabelini et al.:

> Lucas Tabelini et al., *Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection*, CVPR 2021.

If you use the original LaneATT codebase, please cite their paper.

## Research

This repository is developed at Columbia University.
