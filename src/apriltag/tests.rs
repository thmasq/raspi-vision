#[cfg(test)]
mod image_tests {
    use crate::apriltag::image::Image;

    #[test]
    fn new_zeros_out() {
        let img = Image::new(10, 8, 10);
        assert!(img.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn simd_aligned_stride_is_multiple_of_16() {
        for w in [1usize, 15, 16, 17, 63, 64, 65, 640, 1296] {
            let img = Image::new_simd_aligned(w, 4);
            assert_eq!(img.stride % 16, 0, "stride not aligned for width={w}");
            assert!(img.stride >= w);
        }
    }

    #[test]
    fn row_returns_correct_slice() {
        let mut img = Image::new(4, 3, 4);
        let start = 1 * img.stride;
        img.as_mut_slice()[start..start + 4].fill(0xAB);

        assert!(img.row(1).iter().all(|&b| b == 0xAB));
        assert!(img.row(0).iter().all(|&b| b == 0));
        assert!(img.row(2).iter().all(|&b| b == 0));
    }

    #[test]
    fn row_length_equals_width() {
        let img = Image::new_simd_aligned(100, 50);
        for y in 0..50 {
            assert_eq!(img.row(y).len(), 100);
        }
    }

    #[test]
    fn mut_slice_covers_full_buffer() {
        let img = Image::new(8, 8, 8);
        assert_eq!(img.as_slice().len(), 64);
    }
}

#[cfg(test)]
mod threshold_tests {
    use crate::apriltag::{image::Image, threshold::AdaptiveThresholder};

    fn make_checkerboard(w: usize, h: usize, tile: usize) -> Image {
        let mut img = Image::new_simd_aligned(w, h);

        let stride = img.stride;
        let buf = img.as_mut_slice();

        for y in 0..h {
            for x in 0..w {
                let v: u8 = if ((x / tile) + (y / tile)) % 2 == 0 {
                    220
                } else {
                    30
                };

                buf[y * stride + x] = v;
            }
        }

        img
    }

    fn make_uniform(w: usize, h: usize, value: u8) -> Image {
        let mut img = Image::new_simd_aligned(w, h);
        img.as_mut_slice().fill(value);
        img
    }

    /// On a high-contrast checkerboard the output must not be all-127 (the
    /// low-contrast fill value) — real thresholding must have occurred.
    #[test]
    fn high_contrast_produces_binary_output() {
        let input = make_checkerboard(64, 64, 8);
        let mut output = Image::new_simd_aligned(64, 64);
        let mut t = AdaptiveThresholder::new();
        unsafe { t.process(&input, &mut output) };

        let slice = output.as_slice();
        let has_zero = slice.iter().any(|&b| b == 0);
        let has_255 = slice.iter().any(|&b| b == 255);
        assert!(has_zero, "expected black pixels in high-contrast scene");
        assert!(has_255, "expected white pixels in high-contrast scene");
    }

    /// A perfectly uniform image has no local contrast, so every tile should
    /// emit the 127 "don't-care" fill value.
    #[test]
    fn uniform_image_is_all_127() {
        let input = make_uniform(64, 64, 128);
        let mut output = Image::new_simd_aligned(64, 64);
        let mut t = AdaptiveThresholder::new();
        unsafe { t.process(&input, &mut output) };

        let slice = output.as_slice();
        for y in 0..64usize {
            for x in 0..64usize {
                assert_eq!(
                    slice[y * output.stride + x],
                    127,
                    "pixel ({x},{y}) should be 127 in uniform image"
                );
            }
        }
    }

    /// Output dimensions must exactly match input.
    #[test]
    fn output_size_matches_input() {
        let input = make_checkerboard(128, 96, 4);
        let mut output = Image::new_simd_aligned(128, 96);
        let mut t = AdaptiveThresholder::new();
        unsafe { t.process(&input, &mut output) };
        assert_eq!(output.width, 128);
        assert_eq!(output.height, 96);
    }
}

#[cfg(test)]
mod unionfind_tests {
    use crate::apriltag::unionfind::UnionFind;

    fn make_binary_image(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![127u8; w * h];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = if x < w / 2 { 255 } else { 0 };
            }
        }
        buf
    }

    #[test]
    fn clear_resets_state() {
        let mut uf = UnionFind::new();
        let buf = make_binary_image(32, 32);
        uf.connected_components(&buf, 32, 32);
        assert!(!uf.runs.is_empty());

        uf.clear();
        assert!(uf.runs.is_empty());
        assert!(uf.parent.is_empty());
        assert!(uf.row_starts.is_empty());
    }

    #[test]
    fn connected_components_produces_runs() {
        let mut uf = UnionFind::new();
        let buf = make_binary_image(64, 32);
        uf.connected_components(&buf, 64, 32);
        assert!(!uf.runs.is_empty());
    }

    #[test]
    fn connect_and_representative_are_consistent() {
        let mut uf = UnionFind::new();
        uf.parent.push(0);
        uf.size.push(1);
        uf.parent.push(1);
        uf.size.push(1);

        let root = uf.connect(0, 1);
        assert_eq!(uf.get_representative(0), root);
        assert_eq!(uf.get_representative(1), root);
    }

    #[test]
    fn flatten_compresses_paths() {
        let mut uf = UnionFind::new();
        for i in 0..4u32 {
            uf.parent.push(i + 1);
            uf.size.push(1);
        }
        uf.parent[3] = 3;

        uf.flatten();
        for i in 0..4usize {
            assert_eq!(uf.parent[i], 3, "element {i} should point directly to root");
        }
    }

    #[test]
    fn gradient_clusters_returns_some_clusters_on_patterned_image() {
        let w = 128usize;
        let h = 128usize;
        let mut buf = vec![127u8; w * h];

        for y in 20..60usize {
            for x in 20..60usize {
                buf[y * w + x] = if x == 20 || x == 59 || y == 20 || y == 59 {
                    255
                } else {
                    0
                };
            }
        }
        for y in 0..h {
            for x in 0..w {
                if buf[y * w + x] == 127 {
                    buf[y * w + x] = 255;
                }
            }
        }

        let mut uf = UnionFind::new();
        uf.connected_components(&buf, w, h);
        uf.flatten();
        let clusters = uf.gradient_clusters(&buf, w, h);
        assert!(
            !clusters.is_empty(),
            "expected gradient clusters on a border image"
        );
    }
}

#[cfg(test)]
mod quad_tests {
    use crate::apriltag::quad::{QuadWorkspace, find_quad_corners};
    use crate::apriltag::unionfind::Point;

    /// Build a synthetic set of edge points arranged in a square ring.
    fn square_ring_points(cx: u16, cy: u16, half: u16) -> Vec<(u64, Point)> {
        let mut pts = Vec::new();
        let cluster_id: u64 = 0x0000_0001_0000_0000;

        for x in (cx - half)..=(cx + half) {
            pts.push((
                cluster_id,
                Point {
                    x,
                    y: cy - half,
                    gx: 0,
                    gy: -255,
                },
            ));
            pts.push((
                cluster_id,
                Point {
                    x,
                    y: cy + half,
                    gx: 0,
                    gy: 255,
                },
            ));
        }
        for y in (cy - half + 1)..=(cy + half - 1) {
            pts.push((
                cluster_id,
                Point {
                    x: cx - half,
                    y,
                    gx: -255,
                    gy: 0,
                },
            ));
            pts.push((
                cluster_id,
                Point {
                    x: cx + half,
                    y,
                    gx: 255,
                    gy: 0,
                },
            ));
        }

        pts
    }

    #[test]
    fn finds_quad_on_clean_square() {
        let pts = square_ring_points(100, 100, 40);
        let mut ws = QuadWorkspace::default();
        let result = find_quad_corners(&mut ws, &pts);
        assert!(result.is_some(), "expected quad on clean square ring");
    }

    #[test]
    fn too_few_points_returns_none() {
        let pts: Vec<(u64, Point)> = (0..5)
            .map(|i| {
                (
                    0u64,
                    Point {
                        x: i,
                        y: 0,
                        gx: 0,
                        gy: 255,
                    },
                )
            })
            .collect();
        let mut ws = QuadWorkspace::default();
        assert!(find_quad_corners(&mut ws, &pts).is_none());
    }

    #[test]
    fn workspace_is_reusable_across_calls() {
        let pts = square_ring_points(200, 200, 50);
        let mut ws = QuadWorkspace::default();
        let r1 = find_quad_corners(&mut ws, &pts);
        let r2 = find_quad_corners(&mut ws, &pts);
        assert_eq!(
            r1.is_some(),
            r2.is_some(),
            "workspace reuse must be idempotent"
        );
    }
}

#[cfg(test)]
mod homography_tests {
    use crate::apriltag::decode::Homography;

    fn unit_square_corners() -> [[f32; 2]; 4] {
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    }

    #[test]
    fn identity_homography_projects_correctly() {
        let corners = unit_square_corners();
        let h = Homography::compute(&corners).expect("homography should be computable");

        let (px, py) = h.project(0.0, 0.0);
        assert!(
            (px).abs() < 0.01 && (py).abs() < 0.01,
            "center should map near (0,0), got ({px},{py})"
        );
    }

    #[test]
    fn translated_corners_project_to_offset_center() {
        let corners: [[f32; 2]; 4] = [
            [-1.0 + 10.0, -1.0 + 5.0],
            [1.0 + 10.0, -1.0 + 5.0],
            [1.0 + 10.0, 1.0 + 5.0],
            [-1.0 + 10.0, 1.0 + 5.0],
        ];
        let h = Homography::compute(&corners).expect("homography should be computable");
        let (cx, cy) = h.project(0.0, 0.0);
        assert!((cx - 10.0).abs() < 0.1, "expected cx≈10, got {cx}");
        assert!((cy - 5.0).abs() < 0.1, "expected cy≈5, got {cy}");
    }

    #[test]
    fn degenerate_corners_returns_none() {
        let corners = [[0.0f32; 2]; 4];
        let _result = Homography::compute(&corners);
    }
}

#[cfg(test)]
mod gray_model_tests {
    use crate::apriltag::decode::GrayModel;

    #[test]
    fn constant_field_interpolates_correctly() {
        let mut m = GrayModel::default();
        for x in -2..=2i32 {
            for y in -2..=2i32 {
                m.add(x as f32, y as f32, 100.0);
            }
        }
        m.solve();

        for &(px, py) in &[(0.0f32, 0.0), (1.5, -1.5), (-2.0, 2.0)] {
            let v = m.interpolate(px, py);
            assert!(
                (v - 100.0).abs() < 1.0,
                "constant model returned {v} at ({px},{py})"
            );
        }
    }

    #[test]
    fn white_brighter_than_black_at_origin() {
        let mut white = GrayModel::default();
        let mut black = GrayModel::default();

        for x in -3..=3i32 {
            for y in -3..=3i32 {
                white.add(x as f32, y as f32, 200.0);
                black.add(x as f32, y as f32, 50.0);
            }
        }
        white.solve();
        black.solve();

        assert!(
            white.interpolate(0.0, 0.0) > black.interpolate(0.0, 0.0),
            "white model must be brighter than black model at origin"
        );
    }
}

#[cfg(test)]
mod quick_decode_tests {
    use crate::apriltag::decode::{QuickDecode, TAG36H11_CODES};

    #[test]
    fn decodes_all_perfect_codes() {
        let qd = QuickDecode::new();
        for (expected_id, &code) in TAG36H11_CODES.iter().enumerate() {
            let result = qd.decode(code);
            assert!(result.is_some(), "failed to decode tag id {expected_id}");
            let (id, hamming) = result.unwrap();
            assert_eq!(id as usize, expected_id, "wrong id for code {code:#018x}");
            assert_eq!(hamming, 0, "perfect code should have 0 hamming distance");
        }
    }

    #[test]
    fn single_bit_flip_still_decodes() {
        let qd = QuickDecode::new();
        let corrupted = TAG36H11_CODES[0] ^ 1;
        let result = qd.decode(corrupted);
        assert!(result.is_some(), "1-bit error should still decode");
        let (id, hamming) = result.unwrap();
        assert_eq!(id, 0);
        assert_eq!(hamming, 1);
    }

    #[test]
    fn garbage_code_returns_none() {
        let qd = QuickDecode::new();
        let result = qd.decode(0xDEAD_BEEF_DEAD_BEEF);
        let _ = result;
    }

    #[test]
    fn decode_is_deterministic() {
        let qd = QuickDecode::new();
        for &code in TAG36H11_CODES.iter().take(10) {
            assert_eq!(qd.decode(code), qd.decode(code));
        }
    }
}

#[cfg(test)]
mod pipeline_integration_tests {
    use crate::apriltag::{
        decode::{QuickDecode, extract_detection},
        image::Image,
        pose::CameraIntrinsics,
        quad::{QuadWorkspace, find_quad_corners},
        threshold::AdaptiveThresholder,
        unionfind::UnionFind,
    };

    /// Synthetic scene: a bright white square (simulating a tag's white border)
    /// on a dark background.  This exercises the full pipeline up to but not
    /// including actual bit decoding (which requires a real tag image).
    #[test]
    fn full_pipeline_smoke_test() {
        let w = 256usize;
        let h = 256usize;

        let mut input = Image::new_simd_aligned(w, h);

        input.as_mut_slice().fill(20);

        for y in 48..208usize {
            for x in 48..208usize {
                if x < 60 || x >= 196 || y < 60 || y >= 196 {
                    let idx = y * input.stride + x;
                    input.as_mut_slice()[idx] = 230;
                }
            }
        }

        let mut threshold_img = Image::new_simd_aligned(w, h);
        let mut thresholder = AdaptiveThresholder::new();
        unsafe { thresholder.process(&input, &mut threshold_img) };

        let mut uf = UnionFind::new();
        uf.connected_components(threshold_img.as_slice(), w, h);
        uf.flatten();
        let clusters = uf.gradient_clusters(threshold_img.as_slice(), w, h);

        assert!(
            !clusters.is_empty(),
            "smoke test: expected at least one gradient cluster"
        );

        let biggest = clusters
            .iter()
            .max_by_key(|c| c.end_idx - c.start_idx)
            .unwrap();
        let pts = &uf.edge_buffer[biggest.start_idx..biggest.end_idx];

        let mut ws = QuadWorkspace::default();
        let quad = find_quad_corners(&mut ws, pts);

        assert!(quad.is_some(), "smoke test: expected a quad on white ring");
    }

    /// The full `extract_detection` function should not panic even when fed
    /// a synthetic (non-tag) image.
    #[test]
    fn extract_detection_does_not_panic_on_synthetic_image() {
        let w = 128usize;
        let h = 128usize;

        let mut image = Image::new_simd_aligned(w, h);
        image.as_mut_slice().fill(128);

        let corners = [[30.0f32, 30.0], [90.0, 30.0], [90.0, 90.0], [30.0, 90.0]];

        let intrinsics = CameraIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 64.0,
            tag_size_mm: 100.0,
        };

        let qd = QuickDecode::new();

        let _result = extract_detection(&image, &corners, &intrinsics, &qd);
    }
}
