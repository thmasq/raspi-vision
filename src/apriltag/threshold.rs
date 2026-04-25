use crate::apriltag::image::Image;

const TILE_SIZE: usize = 4;
const MIN_CONTRAST: u8 = 30;

/// Performs adaptive thresholding on the input image, writing the binarized
/// result (0 for black, 255 for white) to the output image.
pub fn process(input: &Image, output: &mut Image) {
    debug_assert_eq!(input.width, output.width);
    debug_assert_eq!(input.height, output.height);

    let tiles_x = input.width / TILE_SIZE;
    let tiles_y = input.height / TILE_SIZE;

    // Standard Vec handles heap allocation safely on Linux
    let mut tile_mins = Vec::with_capacity(tiles_x * tiles_y);
    let mut tile_maxs = Vec::with_capacity(tiles_x * tiles_y);

    // =========================================================================
    // PHASE 1: Localized Min/Max Extraction (Auto-Vectorized by LLVM)
    // =========================================================================
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x_start = tx * TILE_SIZE;

            let mut min = 255u8;
            let mut max = 0u8;

            for i in 0..TILE_SIZE {
                let y = ty * TILE_SIZE + i;
                let row_slice = &input.row(y)[x_start..x_start + TILE_SIZE];

                // The compiler unrolls this loop and vectorizes it automatically
                for &val in row_slice {
                    min = min.min(val);
                    max = max.max(val);
                }
            }

            tile_mins.push(min);
            tile_maxs.push(max);
        }
    }

    // =========================================================================
    // PHASE 2: 3x3 Neighborhood Smoothing & Binarization
    // =========================================================================
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let min_tx = tx.saturating_sub(1);
            let max_tx = (tx + 1).min(tiles_x - 1);
            let min_ty = ty.saturating_sub(1);
            let max_ty = (ty + 1).min(tiles_y - 1);

            // Explicit type annotations (u8) fix the ambiguous numeric type error
            let mut local_min: u8 = 255;
            let mut local_max: u8 = 0;

            for ny in min_ty..=max_ty {
                for nx in min_tx..=max_tx {
                    let idx = ny * tiles_x + nx;
                    local_min = local_min.min(tile_mins[idx]);
                    local_max = local_max.max(tile_maxs[idx]);
                }
            }

            let contrast = local_max.saturating_sub(local_min);
            let thresh = local_min as u16 + (contrast as u16 >> 1);

            let x_start = tx * TILE_SIZE;

            for i in 0..TILE_SIZE {
                let y = ty * TILE_SIZE + i;
                let in_row = &input.row(y)[x_start..x_start + TILE_SIZE];
                let out_row = &mut output.row_mut(y)[x_start..x_start + TILE_SIZE];

                for j in 0..TILE_SIZE {
                    if contrast < MIN_CONTRAST {
                        out_row[j] = 0;
                    } else {
                        out_row[j] = if (in_row[j] as u16) < thresh { 255 } else { 0 };
                    }
                }
            }
        }
    }
}
