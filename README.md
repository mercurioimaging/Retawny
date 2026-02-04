# ReTawny V2

Stitches orthophoto tiles (georeferenced TIFFs) into a single PNG using **OpenCV MultiBandBlender**, with per-tile mask support and configurable feathering.

## Usage

```bash
retawny <input_folder> <output.png> [num_bands] [feather_radius]
```

- **input_folder**: folder containing `.tif` / `.tfw` (and optionally `*_mask.png`)
- **output.png**: output PNG path
- **num_bands**: number of bands for MultiBandBlender (default 14, 0–50)
- **feather_radius**: feathering radius in pixels (default 512.0)

## Build (macOS)

```bash
./build.sh
```

Uses qmake + make. The script removes the AGL framework from the generated Makefile (incompatible with recent macOS SDKs).

## Changelog / Evolution

- **CLI**: optional `num_bands` and `feather_radius` command-line arguments.
- **Per-tile masks**: loads `*_mask.png` for each image (black = usable, white = masked); otherwise falls back to magenta pixels.
- **Feathering**: distance transform from masked regions and borders, normalized by `feather_radius`; sharp mode when `feather_radius <= 1`.
- **Memory / OpenCL**: OpenCL is disabled automatically when estimated VRAM usage > 4 GB or when `num_bands > 5` to avoid memory spikes.
- **OrthoLoader**: uses `Orthophotomosaic.tfw` and `MTDOrtho.xml` for canvas size and resolution; strict TFW parsing (rotation/resolution); image/mask path resolution; tile and mask unload after processing.
- **macOS build**: `build.sh` strips `-framework AGL` and AGL includes from the qmake-generated Makefile.
- **Statistics**: per-step timing, total time, peak memory (getrusage).
