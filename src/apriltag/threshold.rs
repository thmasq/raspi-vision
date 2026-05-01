use crate::apriltag::image::Image;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vbslq_u8, vcgtq_u8, vdupq_n_u8, vdupq_n_u32, vextq_u8, vgetq_lane_u8, vld1q_u8, vmaxq_u8,
    vminq_u8, vreinterpretq_u8_u32, vsetq_lane_u32, vst1q_u8,
};
#[cfg(target_arch = "aarch64")]
use core::arch::asm;

const TILE_SIZE: usize = 4;
const MIN_CONTRAST: u8 = 30;

pub struct AdaptiveThresholder {
    tile_mins: Vec<u8>,
    tile_maxs: Vec<u8>,
}

impl Default for AdaptiveThresholder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveThresholder {
    pub fn new() -> Self {
        Self {
            tile_mins: Vec::new(),
            tile_maxs: Vec::new(),
        }
    }

    /// Performs adaptive thresholding on the input image, writing the binarized
    /// result (0 for black, 255 for white) to the output image.
    #[target_feature(enable = "neon")]
    pub unsafe fn process(&mut self, input: &Image, output: &mut Image) {
        debug_assert_eq!(input.width, output.width);
        debug_assert_eq!(input.height, output.height);

        let tiles_x = input.width / TILE_SIZE;
        let tiles_y = input.height / TILE_SIZE;
        let total_tiles = tiles_x * tiles_y;

        self.tile_mins.clear();
        self.tile_mins.resize(total_tiles, 255u8);
        self.tile_maxs.clear();
        self.tile_maxs.resize(total_tiles, 0u8);

        let tile_mins = &mut self.tile_mins;
        let tile_maxs = &mut self.tile_maxs;

        // =========================================================================
        // PHASE 1: Localized Min/Max Extraction
        // =========================================================================
        for ty in 0..tiles_y {
            let y_start = ty * TILE_SIZE;
            let row_mins = &mut tile_mins[ty * tiles_x..(ty + 1) * tiles_x];
            let row_maxs = &mut tile_maxs[ty * tiles_x..(ty + 1) * tiles_x];

            let mut tx = 0;

            while tx + 4 <= tiles_x {
                let x_start = tx * TILE_SIZE;

                let mut min_vec = vdupq_n_u8(255);
                let mut max_vec = vdupq_n_u8(0);

                for i in 0..TILE_SIZE {
                    let y = y_start + i;

                    let prefetch_ptr = input.row(y).as_ptr().wrapping_add(x_start + 64);

                    unsafe {
                        asm!(
                            "prfm pldl1keep, [{}]",
                            in(reg) prefetch_ptr,
                            options(readonly, nostack, preserves_flags)
                        );
                    }

                    let row_ptr = unsafe { input.row(y).get_unchecked(x_start..).as_ptr() };
                    let vals = unsafe { vld1q_u8(row_ptr) };

                    min_vec = vminq_u8(min_vec, vals);
                    max_vec = vmaxq_u8(max_vec, vals);
                }

                let min_shift1 = vextq_u8(min_vec, min_vec, 1);
                let min_step1 = vminq_u8(min_vec, min_shift1);
                let min_shift2 = vextq_u8(min_step1, min_step1, 2);
                let min_final = vminq_u8(min_step1, min_shift2);

                unsafe {
                    *row_mins.get_unchecked_mut(tx) = vgetq_lane_u8(min_final, 0);
                    *row_mins.get_unchecked_mut(tx + 1) = vgetq_lane_u8(min_final, 4);
                    *row_mins.get_unchecked_mut(tx + 2) = vgetq_lane_u8(min_final, 8);
                    *row_mins.get_unchecked_mut(tx + 3) = vgetq_lane_u8(min_final, 12);
                }

                let max_shift1 = vextq_u8(max_vec, max_vec, 1);
                let max_step1 = vmaxq_u8(max_vec, max_shift1);
                let max_shift2 = vextq_u8(max_step1, max_step1, 2);
                let max_final = vmaxq_u8(max_step1, max_shift2);

                unsafe {
                    *row_maxs.get_unchecked_mut(tx) = vgetq_lane_u8(max_final, 0);
                    *row_maxs.get_unchecked_mut(tx + 1) = vgetq_lane_u8(max_final, 4);
                    *row_maxs.get_unchecked_mut(tx + 2) = vgetq_lane_u8(max_final, 8);
                    *row_maxs.get_unchecked_mut(tx + 3) = vgetq_lane_u8(max_final, 12);
                }

                tx += 4;
            }

            while tx < tiles_x {
                let x_start = tx * TILE_SIZE;
                let mut min = 255u8;
                let mut max = 0u8;

                for i in 0..TILE_SIZE {
                    let y = y_start + i;
                    for &val in unsafe { input.row(y).get_unchecked(x_start..x_start + TILE_SIZE) }
                    {
                        min = min.min(val);
                        max = max.max(val);
                    }
                }
                *unsafe { row_mins.get_unchecked_mut(tx) } = min;
                *unsafe { row_maxs.get_unchecked_mut(tx) } = max;
                tx += 1;
            }
        }

        // =========================================================================
        // PHASE 2: 3x3 Neighborhood Smoothing & Binarization
        // =========================================================================
        let width = input.width;
        let out_slice = output.as_mut_slice();

        for ty in 0..tiles_y {
            let min_ty = ty.saturating_sub(1);
            let max_ty = (ty + 1).min(tiles_y - 1);
            let out_row_base = ty * TILE_SIZE * width;

            let mut tx = 0;

            while tx + 4 <= tiles_x {
                let mut thresh_u32x4 = vdupq_n_u32(0);
                let mut mask_u32x4 = vdupq_n_u32(0);
                let mut all_low_contrast = true;

                for j in 0..4 {
                    let current_tx = tx + j;
                    let min_tx = current_tx.saturating_sub(1);
                    let max_tx = (current_tx + 1).min(tiles_x - 1);

                    let mut local_min = 255u8;
                    let mut local_max = 0u8;

                    for ny in min_ty..=max_ty {
                        for nx in min_tx..=max_tx {
                            let idx = ny * tiles_x + nx;
                            local_min = local_min.min(*unsafe { tile_mins.get_unchecked(idx) });
                            local_max = local_max.max(*unsafe { tile_maxs.get_unchecked(idx) });
                        }
                    }

                    let contrast = local_max.saturating_sub(local_min);
                    if contrast >= MIN_CONTRAST {
                        all_low_contrast = false;
                    }

                    let thresh = local_min + (contrast >> 1);
                    let low_contrast = if contrast < MIN_CONTRAST {
                        0xFF_u8
                    } else {
                        0x00_u8
                    };

                    let thresh_32 = u32::from(thresh) * 0x0101_0101;
                    let mask_32 = u32::from(low_contrast) * 0x0101_0101;

                    match j {
                        0 => {
                            thresh_u32x4 = vsetq_lane_u32(thresh_32, thresh_u32x4, 0);
                            mask_u32x4 = vsetq_lane_u32(mask_32, mask_u32x4, 0);
                        }
                        1 => {
                            thresh_u32x4 = vsetq_lane_u32(thresh_32, thresh_u32x4, 1);
                            mask_u32x4 = vsetq_lane_u32(mask_32, mask_u32x4, 1);
                        }
                        2 => {
                            thresh_u32x4 = vsetq_lane_u32(thresh_32, thresh_u32x4, 2);
                            mask_u32x4 = vsetq_lane_u32(mask_32, mask_u32x4, 2);
                        }
                        3 => {
                            thresh_u32x4 = vsetq_lane_u32(thresh_32, thresh_u32x4, 3);
                            mask_u32x4 = vsetq_lane_u32(mask_32, mask_u32x4, 3);
                        }
                        _ => unreachable!(),
                    }
                }

                let x_start = tx * TILE_SIZE;

                if all_low_contrast {
                    let fill_127_vec = vdupq_n_u8(127);
                    for i in 0..TILE_SIZE {
                        let out_idx = out_row_base + i * width + x_start;
                        unsafe {
                            vst1q_u8(
                                out_slice.get_unchecked_mut(out_idx..).as_mut_ptr(),
                                fill_127_vec,
                            );
                        };
                    }
                } else {
                    let thresh_vec = vreinterpretq_u8_u32(thresh_u32x4);
                    let mask_vec = vreinterpretq_u8_u32(mask_u32x4);
                    let fill_127_vec = vdupq_n_u8(127);

                    for i in 0..TILE_SIZE {
                        let y = ty * TILE_SIZE + i;

                        let in_ptr = unsafe { input.row(y).get_unchecked(x_start..).as_ptr() };
                        let in_vals = unsafe { vld1q_u8(in_ptr) };

                        let binarized = vcgtq_u8(in_vals, thresh_vec);
                        let final_out = vbslq_u8(mask_vec, fill_127_vec, binarized);

                        let out_idx = out_row_base + i * width + x_start;
                        unsafe {
                            vst1q_u8(
                                out_slice.get_unchecked_mut(out_idx..).as_mut_ptr(),
                                final_out,
                            );
                        };
                    }
                }
                tx += 4;
            }

            // =========================================================================
            // TAIL LOOP: Scalar fallback for remaining tiles
            // =========================================================================
            while tx < tiles_x {
                let min_tx = tx.saturating_sub(1);
                let max_tx = (tx + 1).min(tiles_x - 1);

                let mut local_min = 255u8;
                let mut local_max = 0u8;

                for ny in min_ty..=max_ty {
                    for nx in min_tx..=max_tx {
                        let idx = ny * tiles_x + nx;
                        local_min = local_min.min(*unsafe { tile_mins.get_unchecked(idx) });
                        local_max = local_max.max(*unsafe { tile_maxs.get_unchecked(idx) });
                    }
                }

                let contrast = local_max.saturating_sub(local_min);
                let thresh = local_min + (contrast >> 1);
                let x_start = tx * TILE_SIZE;

                if contrast < MIN_CONTRAST {
                    for i in 0..TILE_SIZE {
                        let out_idx = out_row_base + i * width + x_start;
                        let out_row =
                            unsafe { out_slice.get_unchecked_mut(out_idx..out_idx + TILE_SIZE) };
                        out_row.fill(127);
                    }
                } else {
                    for i in 0..TILE_SIZE {
                        let y = ty * TILE_SIZE + i;
                        let in_row =
                            unsafe { input.row(y).get_unchecked(x_start..x_start + TILE_SIZE) };
                        let out_idx = out_row_base + i * width + x_start;
                        let out_row =
                            unsafe { out_slice.get_unchecked_mut(out_idx..out_idx + TILE_SIZE) };

                        for j in 0..TILE_SIZE {
                            *unsafe { out_row.get_unchecked_mut(j) } =
                                if *unsafe { in_row.get_unchecked(j) } > thresh {
                                    255
                                } else {
                                    0
                                };
                        }
                    }
                }
                tx += 1;
            }
        }
    }
}
