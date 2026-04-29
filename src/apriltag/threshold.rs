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

    let mut tile_mins = vec![255u8; tiles_x * tiles_y];
    let mut tile_maxs = vec![0u8; tiles_x * tiles_y];

    // =========================================================================
    // PHASE 1: Localized Min/Max Extraction
    // =========================================================================
    for (ty, (mins_row, maxs_row)) in tile_mins
        .chunks_exact_mut(tiles_x)
        .zip(tile_maxs.chunks_exact_mut(tiles_x))
        .enumerate()
    {
        let y_start = ty * TILE_SIZE;

        for tx in 0..tiles_x {
            let x_start = tx * TILE_SIZE;

            let mut min = 255u8;
            let mut max = 0u8;

            for i in 0..TILE_SIZE {
                let y = y_start + i;
                let row_slice = &input.row(y)[x_start..x_start + TILE_SIZE];

                for &val in row_slice {
                    min = min.min(val);
                    max = max.max(val);
                }
            }

            mins_row[tx] = min;
            maxs_row[tx] = max;
        }
    }

    // =========================================================================
    // PHASE 2: 3x3 Neighborhood Smoothing & Binarization
    // =========================================================================
    let out_slice = output.as_mut_slice();
    let width = input.width;
    let row_stride = width * TILE_SIZE;

    for (ty, out_band) in out_slice.chunks_exact_mut(row_stride).enumerate() {
        let min_ty = ty.saturating_sub(1);
        let max_ty = (ty + 1).min(tiles_y - 1);

        for tx in 0..tiles_x {
            let min_tx = tx.saturating_sub(1);
            let max_tx = (tx + 1).min(tiles_x - 1);

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
            let thresh = u16::from(local_min) + (u16::from(contrast) >> 1);
            let x_start = tx * TILE_SIZE;

            if contrast < MIN_CONTRAST {
                for i in 0..TILE_SIZE {
                    let out_row_start = i * width + x_start;
                    let out_row = &mut out_band[out_row_start..out_row_start + TILE_SIZE];
                    out_row.fill(127);
                }
            } else {
                for i in 0..TILE_SIZE {
                    let y = ty * TILE_SIZE + i;
                    let in_row = &input.row(y)[x_start..x_start + TILE_SIZE];
                    let out_row_start = i * width + x_start;
                    let out_row = &mut out_band[out_row_start..out_row_start + TILE_SIZE];

                    for j in 0..TILE_SIZE {
                        out_row[j] = if u16::from(in_row[j]) > thresh {
                            255
                        } else {
                            0
                        };
                    }
                }
            }
        }
    }
}
