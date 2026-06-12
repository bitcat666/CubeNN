<p align="center">
  <img src="https://img.shields.io/badge/Swift-5.5%2B-orange" alt="Swift">
  <img src="https://img.shields.io/badge/Metal-GPU-blue" alt="Metal">
  <img src="https://img.shields.io/badge/Size-161KB-green" alt="Size">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

# CubeNN

**A 14-face geometric neural architecture. 161KB. Pure Swift+Metal. Zero dependencies.**

CubeNN reimagines the neural network as a Rubik's cube. Instead of attention over flat token sequences, neurons are organized across 14 cube faces — 6 standard faces for computation, 8 diagonal X-faces for cross-face communication. Block-sparse attention achieves O(B² log N) scaling, and a tree+chain cascade enables parallel exploration with infinite-depth refinement.

---

## Why This Exists

Modern neural networks are 2D by convention, not by necessity.

A Transformer flattens everything into a sequence and computes attention between every pair of tokens — O(n²), billions of FLOP per layer. It works, but it's brute force.

What if the geometry of the model *matched* the geometry of the problem? What if neurons weren't organized in a line, but on the faces of a cube, with diagonal pathways between them?

That's CubeNN.

---

## Architecture
14 total faces per cube: 6 standard faces → block-sparse attention + FFN, 8 X-faces → diagonal cross-face bridges.

### Block-Sparse Hierarchical Attention

| Scale | Full Attention (同等神经元) | CubeNN | Reduction |
|-------|---------------------------|--------|-----------|
| 6K 神经元 (N=32) | 201.6M FLOP/层 | **0.37M** | 544× |
| 25K 神经元 (N=64) | 3.2B FLOP/层 | **2.2M** | 1,454× |
| 98K 神经元 (N=128) | 137B FLOP/层 | **13M** | 10,500× |
| 393K 神经元 (N=256) | 2.2T FLOP/层 | **52M** | 42,300× |

At 98K neurons, full attention needs **38GB** just for the score matrix. CubeNN uses **467KB**.

### Cube Cascade: Tree + Chain Inference

```
Tree Phase (Exploration):        Chain Phase (Refinement):
  Root cube                       Best leaf selected
  ├→ 8 children (parallel)       → Refine → Refine → Refine
  ├→ 8 children                   ...to convergence (3-5 steps)
  └→ 8 children                   
  73 cubes explored simultaneously
```

No other ML architecture does this.

---

## Performance

**AMD Radeon RX Vega 64** (2017, 14nm, 12.6 TFLOPS FP32):

| Config | Neurons | Time | Notes |
|--------|---------|------|-------|
| N=32, tree | 6,144 × 73 | **108ms** | Warmup + 6x parallel, ~676 FPS equiv |
| N=64, tree | 24,576 × 21 | 817ms | Block-sparse |
| N=128 | 98,304 | **28ms** | ~36 FPS on Vega 64 |

Estimated on modern hardware (N=128):

| GPU | Est. Time | FPS |
|-----|-----------|-----|
| RTX 4090 | ~4ms | 232 |
| Apple M4 Pro | ~16ms | **64** |
| Apple M2 Ultra | ~13ms | 79 |

---

## Key Innovations

1. **14-Face Cube Geometry** — Neurons organized across 6 standard + 8 diagonal faces. X-faces provide cross-face communication without quadratic overhead.

2. **Two-Level Block-Sparse Attention** — Coarse global attention between blocks (64² = 4K scores) + fine local attention within blocks. O(B² log N).

3. **Cube Cascade** — Tree-structured parallel exploration (BFS) selects optimal state for chain-based infinite refinement. 73 cubes run in a single GPU command buffer.

4. **Pure GPU Execution** — Single Metal command buffer, 10,000+ dispatches, zero CPU-GPU synchronization during inference.

5. **Progressive Curriculum Inference** — N=8 warmup seeds N=128 main pass. 2× speedup with no quality loss. "Preview before lecture."

---

## Size

| | CubeNN | PyTorch | TensorFlow |
|---|--------|---------|------------|
| Binary | **161KB** | 800MB | 1.5GB |
| Dependencies | **0** | 200+ wheels | 150+ wheels |
| Language | Swift+Metal | Python+CUDA | Python+CUDA |

---

## Paper

Full paper: [`paper.pdf`](paper.pdf)

```
@misc{cubenn2026,
  title={CubeNN: A Rubik's Cube-Inspired Neural Architecture
         with Block-Sparse Attention and Cascade Inference},
  year={2026}
}
```

---

## Status

Source code is currently private — the architecture is under active development and patent consideration. If you're interested in collaboration, evaluation, or access:

- **Apple Metal / Core ML teams**: The project is built entirely on Swift+Metal and designed for Apple Silicon. See [forum post](https://developer.apple.com/forums/).
- **AMD GPU engineers**: All benchmarks run on Vega 64 (GCN5). We'd love to test on RDNA4/MI300.
- **Researchers**: Contact for paper discussion or benchmark reproduction.

---

*Built by [bitcat666](https://github.com/bitcat666), 2026.*  
*"What if a Rubik's cube was a computer?"*
