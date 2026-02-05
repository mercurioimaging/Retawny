# ReTawny V2

A high-performance image blending tool for creating seamless orthomosaics from multiple georeferenced tiles using OpenCV's MultiBandBlender with dual-mask blending system.

## Features

- **Dual-mask blending** - Separate masks for weight calculation (PC_) and pixel blending (Voronoi)
- **Voronoi mask generation** - Automatic mask creation based on distance to tile centers
- **Minimal ghosting** - Sharp Voronoi boundaries prevent misalignment artifacts
- **Smooth gradients** - PC_ masks provide wide feathering for homogeneous surfaces
- **Multi-band blending** - Configurable number of bands for smooth transitions (default: 14)
- **Memory-efficient** - Automatic OpenCL fallback for large images
- **Georeferencing support** - Via Orthophotomosaic.tfw and MTDOrtho.xml

## Usage

```bash
./retawny.app/Contents/MacOS/retawny <input_folder> <output.png> [num_bands] [feather_radius] [overlap_margin] [use_voronoi] [debug]
```

### Parameters

- `input_folder` - Directory containing TIFF tiles and their TFW files
- `output.png` - Output image path
- `num_bands` - (Optional) Number of bands for MultiBandBlender (default: 14, range: 0-50)
- `feather_radius` - (Optional) Feathering radius for PC_ masks in pixels (default: 512.0)
- `overlap_margin` - (Optional) Voronoi overlap margin in pixels (default: 20.0)
- `use_voronoi` - (Optional) Enable dual-mask blending with Voronoi masks (true/false/1/0/yes/no, default: true)
- `debug` - (Optional) Save weight and blend masks for each tile next to output (debug/true/1, default: false)

### Examples

**Dual-mask blending (default):**
```bash
./retawny.app/Contents/MacOS/retawny ./tiles output.png 12 64 20
```

**PC_ masks only (no Voronoi):**
```bash
./retawny.app/Contents/MacOS/retawny ./tiles output.png 12 512 20 false
```

**Debug mode (save all masks):**
```bash
./retawny.app/Contents/MacOS/retawny ./tiles output.png 12 64 20 true debug
```

## Build (macOS)

```bash
./build.sh
```

Uses qmake + make. Requires Qt 5/6 and OpenCV 4.x with stitching module.

## Voronoi Mask Generation

ReTawny V2 automatically generates optimal blending masks based on Voronoi diagrams:

### Algorithm

1. **Distance calculation** - For each pixel, compute distance to all tile centers
2. **PC_ mask constraints** - Exclude pixels masked in PC_ files (white = invalid)
3. **Gradient generation** - Pixels within `overlap_margin` of a Voronoi frontier receive gradient values (0-255)

### Output

- Saved as `*_voronoi_mask.tif` alongside source tiles
- Format: 8-bit grayscale with gradient
  - 0: Outside tile ownership or masked
  - 1-254: Transition zone (gradient)
  - 255: Full ownership zone
- Reusable across multiple runs

### Benefits

- **Minimal seams** - Transitions only at Voronoi frontiers (~20-40px wide)
- **Better quality** - No wide feathering zones across entire tiles
- **Data validity** - Respects PC_ mask constraints
- **Performance** - Generated masks can be reused

## Input Requirements

### Required

- `*.tif` / `*.tiff` - Georeferenced image tiles
- `*.tfw` - World files for each tile

### Optional

- `Orthophotomosaic.tfw` - Reference world file for canvas georeferencing
- `MTDOrtho.xml` - Metadata file with canvas dimensions
- `PC_*.tif` - Pre-computed masks (black = valid, white = masked)

## Dual-Mask Blending System

ReTawny V2 uses two separate masks per tile for optimal blending quality:

### Weight Mask (PC_)
- **Purpose**: Controls how much weight each tile contributes to the final result
- **Source**: PC_ mask files (pre-computed coverage masks)
- **Processing**: Binarized with feathering applied at borders and masked regions
- **Benefits**: Wide, smooth gradients ensure homogeneous surface appearance

### Blend Mask (Voronoi)
- **Purpose**: Controls actual pixel blending to avoid ghosting
- **Source**: Auto-generated Voronoi masks based on tile centers
- **Processing**: Gradient generated at Voronoi frontiers (overlap_margin width)
- **Benefits**: Sharp boundaries prevent ghosting from tile misalignment

### Why Two Masks?

This dual-mask approach combines the best of both strategies:
- **No ghosting**: Voronoi boundaries prevent double-vision from misaligned tiles
- **Smooth gradients**: PC_ weights provide wide feathering for seamless surfaces
- **Best quality**: Homogeneous appearance without alignment artifacts

To disable Voronoi masks and use PC_ masks only, pass `false` as the 6th parameter.

## Processing Pipeline

1. **Load Metadata** - Parse TFW files, determine tile positions and dimensions
2. **Generate Voronoi Masks** - Create gradient masks respecting PC_ constraints (~4-7s for 12 tiles, skipped if use_voronoi=false)
3. **Prepare Blender** - Initialize DualMaskMultiBandBlender with canvas size
4. **Process Tiles** - Load tiles, build weight and blend masks, feed to blender
5. **Blend** - Combine all tiles using dual-mask multiband blending
6. **Save** - Export final result

## Performance

Typical results for 12 tiles (4576×3088 px each, 24256×12240 canvas):

- Voronoi generation: ~4-7 seconds
- Total processing: ~1-2 minutes
- Peak memory: ~3-6 GB
- Gradient pixels: ~2-3% of total (20px margin)

## Technical Details

### Coordinate System

- Origin: Top-left (0, 0)
- X-axis: Right
- Y-axis: Down (standard image coordinates)

### Memory Management

- Automatic OpenCL disable for large images (>4GB VRAM)
- Tiles loaded/unloaded individually
- PC_ masks loaded once during generation, then released

### Mask Priority

**Weight Masks (PC_):**
1. PC_ mask file (`PC_*.tif`)
2. Magenta pixel detection in source image (fallback)

**Blend Masks (Voronoi):**
1. Generated Voronoi mask (`*_voronoi_mask.tif`) - if use_voronoi=true
2. Falls back to weight mask if use_voronoi=false or Voronoi generation fails

## Notes

- Voronoi masks are always regenerated when use_voronoi=true (ensures consistency with current overlap_margin)
- PC_ masks use feathering (feather_radius parameter) for smooth transitions
- Voronoi masks use their built-in gradient (no additional feathering applied)
- Statistics (time, memory) reported after completion

## Requirements

- Qt 5/6
- OpenCV 4.x with stitching module
- C++11 or later
- macOS (build script removes incompatible AGL framework)
