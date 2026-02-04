# ReTawny V2

A high-performance image blending tool for creating seamless orthomosaics from multiple georeferenced tiles using OpenCV's MultiBandBlender with automatic Voronoi-based mask generation.

## Features

- **Voronoi mask generation** - Automatic mask creation based on distance to tile centers
- **Minimal seam transitions** - Configurable overlap margin (default: 20px) for precise blending
- **PC_ mask integration** - Respects invalid regions from source masks (white = masked, black = valid)
- **Multi-band blending** - Configurable number of bands for smooth transitions (default: 14)
- **Memory-efficient** - Automatic OpenCL fallback for large images
- **Georeferencing support** - Via Orthophotomosaic.tfw and MTDOrtho.xml

## Usage

```bash
./retawny.app/Contents/MacOS/retawny <input_folder> <output.png> [num_bands] [feather_radius] [overlap_margin]
```

### Parameters

- `input_folder` - Directory containing TIFF tiles and their TFW files
- `output.png` - Output image path
- `num_bands` - (Optional) Number of bands for MultiBandBlender (default: 14, range: 0-50)
- `feather_radius` - (Optional) Feathering radius for PC_ masks in pixels (default: 512.0)
- `overlap_margin` - (Optional) Voronoi overlap margin in pixels (default: 20.0)

### Example

```bash
./retawny.app/Contents/MacOS/retawny ./tiles output.png 12 512 20
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

## Processing Pipeline

1. **Load Metadata** - Parse TFW files, determine tile positions and dimensions
2. **Generate Voronoi Masks** - Create gradient masks respecting PC_ constraints (~4-7s for 12 tiles)
3. **Prepare Blender** - Initialize MultiBandBlender with canvas size
4. **Process Tiles** - Load tiles, apply masks, feed to blender
5. **Blend** - Combine all tiles using multiband blending
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

1. Generated Voronoi mask (`*_voronoi_mask.tif`)
2. PC_ mask (used during Voronoi generation)
3. Magenta pixel detection (fallback)

## Notes

- Voronoi masks are always regenerated (even if PC_ masks exist)
- PC_ masks constrain but don't replace Voronoi masks
- `sharp` parameter automatically set for Voronoi masks to avoid double feathering
- Statistics (time, memory) reported after completion

## Requirements

- Qt 5/6
- OpenCV 4.x with stitching module
- C++11 or later
- macOS (build script removes incompatible AGL framework)
