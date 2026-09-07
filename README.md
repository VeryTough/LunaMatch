# LunaMatch

**High-precision computer vision pipeline for Chandrayaan-2 orbital image registration**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-USAC__MAGSAC-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

LunaMatch registers Chandrayaan-2 Terrain Mapping Camera (TMC) imagery across viewing geometries — matching **Nadir (0°)** and **Aft (−25°)** angles by explicitly modeling 3D lunar parallax, rather than assuming a flat surface.

---

## Table of Contents

- [The Problem](#the-problem)
- [Approach](#approach)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Team](#team)

---

## The Problem

Registering orbital imagery of the Moon is hard because the surface is **not planar**. Craters and ridges introduce real elevation relief, so a standard 2D homography — which assumes all matched points lie on a single plane — systematically fails to warp one image onto another when viewing angles differ.

This breaks naive feature-matching pipelines whenever the Nadir and Aft passes need to be fused, since the same crater projects to different pixel offsets depending on its depth, not just its planar position.

## Approach

LunaMatch replaces the flat-plane assumption with a full epipolar-geometry pipeline:

| Stage | What it does |
|---|---|
| **1. Feature extraction** | PyTorch-accelerated **SuperPoint** + **LightGlue** extract and match thousands of deep-learned keypoints across tiled image chunks |
| **2. Epipolar geometry** | A 3D-aware **Fundamental Matrix** verifier (OpenCV `USAC_MAGSAC`) replaces flat-plane RANSAC, correctly accounting for crater depth and camera tilt |
| **3. Adaptive tiling** | Large orbital strips are processed in localized tiles to preserve sub-pixel alignment accuracy |

This gives geometrically consistent matches even where relief and viewing-angle disparity would defeat a 2D-homography-based approach.

## Results

Phase 2 verification metrics on the current pipeline:

| Metric | Result | Target |
|---|---|---|
| Inlier ratio | **73.53%** | > 65% |
| Registration RMSE | **0.6243 px** | < 0.8 px |
| Geometric consensus | Verified 3D stereoscopic lock (`USAC_MAGSAC`) | — |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Find overlapping tiles between Nadir/Aft strips
python src/matching/auto_find_overlap.py

# 3. Verify registration metrics
python tests/test_ransac.py
```

## Project Structure

```
lunamatch/
├── src/
│   └── matching/
│       └── auto_find_overlap.py   # tile-overlap discovery
├── tests/
│   └── test_ransac.py             # RANSAC / MAGSAC verification
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Nadir–Aft registration prototype (SuperPoint + LightGlue + MAGSAC)
- [x] Phase 2 metric verification (inlier ratio, RMSE)
- [ ] Hardware-in-the-loop rover localization validation
- [ ] Extended SIH prototype with sensor fusion on physical rover

## Team

Built for **Smart India Hackathon 2026** — problem statement **SIH26166 (ISRO)**.

- Anuraag Chakraborty
- Vivek Singh Mathur
- Mayank Nitin Sarode
- Priyanshu Bhati
- Prantar Pallab Mazumdar
- Ananya Bisht

**Mentor:** Dr. Avinash Chandra

---

*Part of the LunaMatch project addressing illumination-robust lunar image correspondence and rover localization.*
