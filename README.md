# Wall Crack Detector

Python/PyQt6 implementation of the MAD adaptive-threshold crack detector from
Avendaño, Leander & Karoumi, *Sensors* 2024, 24, 2736.

    pip install -r requirements.txt
    python main.py

## Structure (MVC)
- `models/`      algorithm + state (no Qt): greyscale, per-tile MAD threshold,
                 Laplacian edges, skeleton, perpendicular width measurement, pinhole scale
- `controllers/` wires view signals to models, runs detection in a QThread
- `views/`       PyQt6 widgets only (main window, zoomable viewer)

## Notes
- k <= 2.5: best detection of 0.1 mm cracks, more noise and width overestimation;
  k = 3.5: best width accuracy, but misses thin cracks (paper, Tables 1-3).
- Pixel size = 2*tan(alpha/2)*u / n_w. Default alpha=40.4 deg reproduces the paper's
  0.1 mm/px at 1 m with a 7380 px wide image. Or enter mm/px manually.
- "Remove short blobs" is an optional post-process the paper proposes as future work.
