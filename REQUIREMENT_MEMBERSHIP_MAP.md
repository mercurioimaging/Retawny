# Requirement: Pixel membership map for minimal seam

## Goal

Have a **membership map** for the canvas: for each pixel, assign the tile it **belongs** to, based on **distance to the nearest tile image center**. Use this so that blending happens only in a **narrow band** between two tiles, giving a **small seam** instead of a wide feathered transition.

## Reference (Tawny V3)

In `08_Tawny_V3`, the same idea is implemented in Python:

- **Ownership (Voronoi)**: For each canvas pixel, the owning tile is the one whose **image center** (in world coordinates) is closest. Pixels are only assigned if they lie inside that tile’s bounds and pass the tile mask.
- **Usage**: That ownership map is then used to build per-tile masks and to assemble with feathering only along the seams (see `determine_tile_ownership` and seam/alpha logic in `ortho_corrector_V3.py` and `lib/seam_detection`).

## What is needed in ReTawny V2

1. **Tile centers (canvas coordinates)**  
   For each tile, define its center in canvas pixel coordinates, e.g.  
   `center_x = tile.x + tile_width/2`, `center_y = tile.y + tile_height/2`.  
   This requires tile width and height to be available (e.g. stored in `OrthoLoader::Tile` and set when metadata or image size is known).

2. **Membership map**  
   A canvas-sized array (e.g. one int per pixel) where each pixel stores the **index of the tile** it belongs to. Assignment rule: among tiles that cover that pixel (and where the pixel is valid according to the tile mask), choose the tile whose **center** is closest (Euclidean distance in canvas or world coordinates). So the map is a Voronoi diagram restricted to valid tile coverage.

3. **Use the map for blending**  
   Use the membership map so that:
   - Inside a tile’s region, that tile’s contribution is dominant (e.g. full weight or only a very small transition).
   - Blending (feathering) is applied only in a **small band** along the boundary between two tiles (e.g. a few pixels to a few tens of pixels), instead of a large feathered area (e.g. current 512 px radius).  
   Result: **only a small seam** between two adjacent tiles, consistent with “distance to nearest image center” ownership.

## Summary in one sentence

**ReTawny V2 should compute a per-pixel membership map (which tile owns each pixel, by distance to nearest tile center) and use it so that blending occurs only in a narrow band along tile boundaries, yielding a small seam between tiles.**
