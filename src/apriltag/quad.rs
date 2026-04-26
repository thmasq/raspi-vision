use crate::apriltag::unionfind::{Blob, Segment};

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub slope: f32,
}

/// Stores cumulative moments for O(1) line fitting queries.
/// Equivalent to `struct line_fit_pt` in the C library.
#[derive(Default, Clone, Copy)]
pub struct LineFitPt {
    pub mx: f64,
    pub my: f64,
    pub mxx: f64,
    pub myy: f64,
    pub mxy: f64,
    pub w: f64,
}

/// Extracts the boundary points from a blob's RLE segments and orders them
/// clockwise around the centroid of the blob.
pub fn extract_ordered_boundary(blob: &Blob, segments: &[Segment]) -> Vec<Point> {
    let expected_perimeter = (blob.pixel_count as f32).sqrt() as usize * 4;
    let mut boundary = Vec::with_capacity(expected_perimeter);

    let blob_segments = &segments[blob.start_idx..blob.end_idx];

    for seg in blob_segments {
        let left = Point {
            x: seg.x_start as f32,
            y: seg.y as f32,
            slope: 0.0,
        };
        let right = Point {
            x: seg.x_end as f32,
            y: seg.y as f32,
            slope: 0.0,
        };

        boundary.push(left);
        if seg.x_start != seg.x_end {
            boundary.push(right);
        }
    }

    if boundary.is_empty() {
        return boundary;
    }

    let mut xmax = boundary[0].x;
    let mut xmin = boundary[0].x;
    let mut ymax = boundary[0].y;
    let mut ymin = boundary[0].y;

    for p in &boundary {
        if p.x > xmax {
            xmax = p.x;
        } else if p.x < xmin {
            xmin = p.x;
        }
        if p.y > ymax {
            ymax = p.y;
        } else if p.y < ymin {
            ymin = p.y;
        }
    }

    let cx = (xmin + xmax) * 0.5 + 0.05118;
    let cy = (ymin + ymax) * 0.5 - 0.028581;

    for pt in boundary.iter_mut() {
        let mut dx = pt.x - cx;
        let mut dy = pt.y - cy;

        let dy_gt_0 = dy > 0.0;
        let dx_gt_0 = dx > 0.0;

        let quadrant = match (dy_gt_0, dx_gt_0) {
            (false, false) => -131072.0,
            (false, true) => 0.0,
            (true, false) => 131072.0,
            (true, true) => 65536.0,
        };

        if dy < 0.0 {
            dy = -dy;
            dx = -dx;
        }
        if dx < 0.0 {
            let tmp = dx;
            dx = dy;
            dy = -tmp;
        }
        pt.slope = quadrant + (dy / dx);
    }

    boundary.sort_unstable_by(|a, b| a.slope.partial_cmp(&b.slope).unwrap());
    boundary
}

/// Precomputes cumulative moments. Equivalent to `compute_lfps` in C.
fn compute_lfps(boundary: &[Point]) -> Vec<LineFitPt> {
    let mut lfps = Vec::with_capacity(boundary.len());
    let mut sum_mx = 0.0;
    let mut sum_my = 0.0;
    let mut sum_mxx = 0.0;
    let mut sum_myy = 0.0;
    let mut sum_mxy = 0.0;
    let mut sum_w = 0.0;

    for p in boundary {
        let x = p.x as f64;
        let y = p.y as f64;
        let w = 1.0;

        sum_mx += w * x;
        sum_my += w * y;
        sum_mxx += w * x * x;
        sum_myy += w * y * y;
        sum_mxy += w * x * y;
        sum_w += w;

        lfps.push(LineFitPt {
            mx: sum_mx,
            my: sum_my,
            mxx: sum_mxx,
            myy: sum_myy,
            mxy: sum_mxy,
            w: sum_w,
        });
    }
    lfps
}

/// Fits a line to points [i0, i1] (inclusive) using cumulative moments in O(1).
/// Returns: (Ex, Ey, nx, ny, err, mse)
fn fit_line(lfps: &[LineFitPt], sz: usize, i0: usize, i1: usize) -> (f32, f32, f32, f32, f32, f32) {
    let (mx, my, mxx, myy, mxy, w);
    let n;

    if i0 <= i1 {
        n = (i1 - i0 + 1) as f64;
        let mut tmp_mx = lfps[i1].mx;
        let mut tmp_my = lfps[i1].my;
        let mut tmp_mxx = lfps[i1].mxx;
        let mut tmp_mxy = lfps[i1].mxy;
        let mut tmp_myy = lfps[i1].myy;
        let mut tmp_w = lfps[i1].w;

        if i0 > 0 {
            tmp_mx -= lfps[i0 - 1].mx;
            tmp_my -= lfps[i0 - 1].my;
            tmp_mxx -= lfps[i0 - 1].mxx;
            tmp_mxy -= lfps[i0 - 1].mxy;
            tmp_myy -= lfps[i0 - 1].myy;
            tmp_w -= lfps[i0 - 1].w;
        }
        mx = tmp_mx;
        my = tmp_my;
        mxx = tmp_mxx;
        mxy = tmp_mxy;
        myy = tmp_myy;
        w = tmp_w;
    } else {
        n = (sz - i0 + i1 + 1) as f64;
        mx = lfps[sz - 1].mx - lfps[i0 - 1].mx + lfps[i1].mx;
        my = lfps[sz - 1].my - lfps[i0 - 1].my + lfps[i1].my;
        mxx = lfps[sz - 1].mxx - lfps[i0 - 1].mxx + lfps[i1].mxx;
        mxy = lfps[sz - 1].mxy - lfps[i0 - 1].mxy + lfps[i1].mxy;
        myy = lfps[sz - 1].myy - lfps[i0 - 1].myy + lfps[i1].myy;
        w = lfps[sz - 1].w - lfps[i0 - 1].w + lfps[i1].w;
    }

    let n = n as f32;
    let mx = mx as f32;
    let my = my as f32;
    let mxx = mxx as f32;
    let myy = myy as f32;
    let mxy = mxy as f32;
    let w = w as f32;

    let ex = mx / w;
    let ey = my / w;
    let cxx = mxx / w - ex * ex;
    let cxy = mxy / w - ex * ey;
    let cyy = myy / w - ey * ey;

    let eig_small = 0.5 * (cxx + cyy - ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());
    let eig_large = 0.5 * (cxx + cyy + ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());

    let nx1 = cxx - eig_large;
    let ny1 = cxy;
    let m1 = nx1 * nx1 + ny1 * ny1;

    let nx2 = cxy;
    let ny2 = cyy - eig_large;
    let m2 = nx2 * nx2 + ny2 * ny2;

    let (mut nx, mut ny, m) = if m1 > m2 {
        (nx1, ny1, m1)
    } else {
        (nx2, ny2, m2)
    };

    let length = m.sqrt();
    if length.abs() < 1e-12 {
        nx = 0.0;
        ny = 0.0;
    } else {
        nx /= length;
        ny /= length;
    }

    let err = n * eig_small;
    let mse = eig_small;

    (ex, ey, nx, ny, err, mse)
}

/// Finds the 4 mathematical sub-pixel corners of an AprilTag quad.
/// Replicates `quad_segment_maxima` and line intersections.
pub fn find_quad_corners(boundary: &[Point]) -> Option<[Point; 4]> {
    let sz = boundary.len();
    if sz < 24 {
        return None;
    }

    let ksz = 20.min(sz / 12);
    if ksz < 2 {
        return None;
    }

    let lfps = compute_lfps(boundary);
    let mut errs = vec![0.0; sz];

    for i in 0..sz {
        let (_, _, _, _, err, _) = fit_line(&lfps, sz, (i + sz - ksz) % sz, (i + ksz) % sz);
        errs[i] = err;
    }

    let sigma = 1.0_f32;
    let cutoff = 0.05_f32;
    let mut fsz = (-cutoff.ln() * 2.0 * sigma * sigma).sqrt() as i32 + 1;
    fsz = 2 * fsz + 1;

    let mut f = Vec::with_capacity(fsz as usize);
    for i in 0..fsz {
        let j = (i - fsz / 2) as f32;
        f.push((-j * j / (2.0 * sigma * sigma)).exp());
    }

    let mut y_errs = vec![0.0; sz];
    for iy in 0..sz {
        let mut acc = 0.0;
        for i in 0..fsz {
            let idx = (iy as i32 + i - fsz / 2 + sz as i32) % (sz as i32);
            acc += errs[idx as usize] * f[i as usize];
        }
        y_errs[iy] = acc;
    }
    errs = y_errs;

    let mut maxima = Vec::new();
    let mut maxima_errs = Vec::new();
    for i in 0..sz {
        if errs[i] > errs[(i + 1) % sz] && errs[i] > errs[(i + sz - 1) % sz] {
            maxima.push(i);
            maxima_errs.push(errs[i]);
        }
    }

    if maxima.len() < 4 {
        return None;
    }

    let max_nmaxima = 10;
    if maxima.len() > max_nmaxima {
        let mut indices: Vec<usize> = (0..maxima.len()).collect();
        indices.sort_unstable_by(|&a, &b| maxima_errs[b].partial_cmp(&maxima_errs[a]).unwrap());

        let threshold = maxima_errs[indices[max_nmaxima]];
        let mut best_maxima = Vec::new();
        for i in 0..maxima.len() {
            if maxima_errs[i] > threshold {
                best_maxima.push(maxima[i]);
            }
        }
        maxima = best_maxima;
    }

    let nmaxima = maxima.len();
    let mut best_indices = [0; 4];
    let mut best_error = f32::INFINITY;
    let max_dot = (25.0_f32 * std::f32::consts::PI / 180.0).cos();
    let max_line_fit_mse = 10.0_f32;

    for m0 in 0..(nmaxima.saturating_sub(3)) {
        let i0 = maxima[m0];
        for m1 in (m0 + 1)..(nmaxima.saturating_sub(2)) {
            let i1 = maxima[m1];
            let (_, _, nx01, ny01, err01, mse01) = fit_line(&lfps, sz, i0, i1);
            if mse01 > max_line_fit_mse {
                continue;
            }

            for m2 in (m1 + 1)..(nmaxima.saturating_sub(1)) {
                let i2 = maxima[m2];
                let (_, _, nx12, ny12, err12, mse12) = fit_line(&lfps, sz, i1, i2);
                if mse12 > max_line_fit_mse {
                    continue;
                }

                let dot = nx01 * nx12 + ny01 * ny12;
                if dot.abs() > max_dot {
                    continue;
                }

                for m3 in (m2 + 1)..nmaxima {
                    let i3 = maxima[m3];
                    let (_, _, _, _, err23, mse23) = fit_line(&lfps, sz, i2, i3);
                    if mse23 > max_line_fit_mse {
                        continue;
                    }

                    let (_, _, _, _, err30, mse30) = fit_line(&lfps, sz, i3, i0);
                    if mse30 > max_line_fit_mse {
                        continue;
                    }

                    let err = err01 + err12 + err23 + err30;

                    if err < best_error {
                        best_error = err;
                        best_indices = [i0, i1, i2, i3];
                    }
                }
            }
        }
    }

    if best_error == f32::INFINITY {
        return None;
    }
    if best_error / (sz as f32) >= max_line_fit_mse {
        return None;
    }

    let mut lines = [(0.0, 0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        let i0 = best_indices[i];
        let i1 = best_indices[(i + 1) & 3];
        let (ex, ey, nx, ny, _, _) = fit_line(&lfps, sz, i0, i1);
        lines[i] = (ex, ey, nx, ny);
    }

    let mut exact_corners = [Point {
        x: 0.0,
        y: 0.0,
        slope: 0.0,
    }; 4];
    for i in 0..4 {
        let (p0_x, p0_y, n0_x, n0_y) = lines[i];
        let (p1_x, p1_y, n1_x, n1_y) = lines[(i + 1) & 3];

        let a00 = n0_y;
        let a01 = -n1_y;
        let a10 = -n0_x;
        let a11 = n1_x;

        let b0 = -p0_x + p1_x;
        let b1 = -p0_y + p1_y;

        let det = a00 * a11 - a10 * a01;
        if det.abs() < 0.001 {
            return None;
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;
        let l0 = w00 * b0 + w01 * b1;

        exact_corners[i] = Point {
            x: (p0_x + l0 * a00) as f32,
            y: (p0_y + l0 * a10) as f32,
            slope: 0.0,
        };
    }

    Some(exact_corners)
}
