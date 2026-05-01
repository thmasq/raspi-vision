#![allow(clippy::inline_always)]

use crate::apriltag::{image::Image, unionfind::Point};

const RANGE: f32 = 2.0;
const STEPS_PER_UNIT: usize = 4;
const STEP_LENGTH: f32 = 1.0 / STEPS_PER_UNIT as f32;
const MAX_STEPS: usize = 2 * STEPS_PER_UNIT * RANGE as usize + 1;
const DELTA: f32 = 0.5;
const GRANGE: f32 = 1.0;

// Minimum number of pixels required for a cluster to be considered.
// Increase if tiny / distant tags are not important.
const MIN_CLUSTER_PIXELS: usize = 24;

// Bounding box size limits.
const MIN_TAG_DIMENSION: u16 = 12;
const MAX_TAG_DIMENSION: u16 = 1200;

// Maximum allowed aspect ratio.
// Example: 5 means up to 5:1 or 1:5.
const MAX_ASPECT_RATIO: u16 = 5;

// Density heuristic divisor.
// Lower = stricter density requirement.
const MIN_DENSITY_DIVISOR: u16 = 3;

// Empirical center offsets.
const CENTER_X_OFFSET: f64 = 0.05118;
const CENTER_Y_OFFSET: f64 = -0.028_581;

#[derive(Debug, Clone, Copy)]
struct FitPoint {
    pub x: f64,
    pub y: f64,
    pub gx: f64,
    pub gy: f64,
    pub slope: f64,
}

/// Stores cumulative moments for O(1) line fitting queries.
/// Equivalent to `struct line_fit_pt` in the C library.
#[derive(Default, Clone, Copy)]
struct LineFitPt {
    pub mx: f64,
    pub my: f64,
    pub mxx: f64,
    pub myy: f64,
    pub mxy: f64,
    pub w: f64,
}

/// Fits a quadrilateral to a point cloud cluster representing a tag boundary.
/// Replicates `fit_quad` and `quad_segment_maxima` from the official C pipeline.
pub fn find_quad_corners(points_slice: &[(u64, Point)]) -> Option<[[f32; 2]; 4]> {
    let pts = prepare_points(points_slice)?;
    let lfps = compute_lfps(&pts);

    let errs = compute_corner_response(&pts, &lfps)?;
    let maxima = detect_corner_candidates(&errs);

    let quad = select_best_quad(&lfps, pts.len(), &maxima)?;
    intersect_quad_lines(&lfps, pts.len(), quad)
}

fn prepare_points(points: &[(u64, Point)]) -> Option<Vec<FitPoint>> {
    if points.len() < MIN_CLUSTER_PIXELS {
        return None;
    }

    // Access the Point struct using .1 from the tuple
    let mut xmin = points[0].1.x;
    let mut xmax = points[0].1.x;
    let mut ymin = points[0].1.y;
    let mut ymax = points[0].1.y;

    for (_, p) in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }

    let width = xmax - xmin;
    let height = ymax - ymin;

    if width < MIN_TAG_DIMENSION
        || height < MIN_TAG_DIMENSION
        || width > MAX_TAG_DIMENSION
        || height > MAX_TAG_DIMENSION
    {
        return None;
    }

    if width > height * MAX_ASPECT_RATIO || height > width * MAX_ASPECT_RATIO {
        return None;
    }

    let approx_perimeter = (width + height) * 2;
    if points.len() < (approx_perimeter / MIN_DENSITY_DIVISOR) as usize {
        return None;
    }

    let cx = (f64::from(xmin) + f64::from(xmax)).mul_add(0.5, CENTER_X_OFFSET);
    let cy = (f64::from(ymin) + f64::from(ymax)).mul_add(0.5, CENTER_Y_OFFSET);

    let mut pts: Vec<FitPoint> = points
        .iter()
        .map(|(_, p)| FitPoint {
            x: f64::from(p.x),
            y: f64::from(p.y),
            gx: f64::from(p.gx),
            gy: f64::from(p.gy),
            slope: 0.0,
        })
        .collect();

    for p in &mut pts {
        let dx = p.x - cx;
        let dy = p.y - cy;
        p.slope = dy.atan2(dx);
    }

    pts.sort_unstable_by(|a, b| a.slope.partial_cmp(&b.slope).unwrap());

    Some(pts)
}

fn compute_corner_response(pts: &[FitPoint], lfps: &[LineFitPt]) -> Option<Vec<f64>> {
    let sz = pts.len();
    let ksz = 20.min(sz / 12);
    if ksz < 2 {
        return None;
    }

    let mut errs = vec![0.0; sz];
    for (i, err_slot) in errs.iter_mut().enumerate().take(sz) {
        let (_, _, _, _, err, _) = fit_line(lfps, sz, (i + sz - ksz) % sz, (i + ksz) % sz);
        *err_slot = err;
    }

    let sigma = 1.0_f64;
    let cutoff = 0.05_f64;
    let mut fsz = (-cutoff.ln() * 2.0 * sigma * sigma).sqrt() as i32 + 1;
    fsz = 2 * fsz + 1;

    let mut f = Vec::with_capacity(fsz as usize);
    for i in 0..fsz {
        let j = f64::from(i - fsz / 2);
        f.push((-j * j / (2.0 * sigma * sigma)).exp());
    }

    let mut y_errs = vec![0.0; sz];

    for (iy, y_err) in y_errs.iter_mut().enumerate().take(sz) {
        let mut acc = 0.0;

        for i in 0..fsz {
            let idx = (iy as i32 + i - fsz / 2 + sz as i32) % (sz as i32);
            acc = errs[idx as usize].mul_add(f[i as usize], acc);
        }

        *y_err = acc;
    }

    Some(y_errs)
}

fn detect_corner_candidates(errs: &[f64]) -> Vec<usize> {
    let sz = errs.len();
    let mut maxima = Vec::new();
    let mut maxima_errs = Vec::new();

    for i in 0..sz {
        if errs[i] > errs[(i + 1) % sz] && errs[i] > errs[(i + sz - 1) % sz] {
            maxima.push(i);
            maxima_errs.push(errs[i]);
        }
    }

    let max_nmaxima = 10;
    if maxima.len() > max_nmaxima {
        let mut indices: Vec<usize> = (0..maxima.len()).collect();
        indices.sort_unstable_by(|&a, &b| maxima_errs[b].partial_cmp(&maxima_errs[a]).unwrap());
        let threshold = maxima_errs[indices[max_nmaxima]];

        maxima = maxima
            .into_iter()
            .zip(maxima_errs)
            .filter(|(_, err)| *err > threshold)
            .map(|(idx, _)| idx)
            .collect();
    }

    maxima
}

fn select_best_quad(lfps: &[LineFitPt], sz: usize, maxima: &[usize]) -> Option<[usize; 4]> {
    if maxima.len() < 4 {
        return None;
    }

    let nmaxima = maxima.len();
    let mut best_indices = [0; 4];
    let mut best_error = f64::INFINITY;
    let max_dot = 25.0_f64.to_radians().cos();
    let max_line_fit_mse = 10.0_f64;

    for m0 in 0..(nmaxima.saturating_sub(3)) {
        let i0 = maxima[m0];
        for m1 in (m0 + 1)..(nmaxima.saturating_sub(2)) {
            let i1 = maxima[m1];
            let (_, _, nx01, ny01, err01, mse01) = fit_line(lfps, sz, i0, i1);
            if mse01 > max_line_fit_mse {
                continue;
            }

            for m2 in (m1 + 1)..(nmaxima.saturating_sub(1)) {
                let i2 = maxima[m2];
                let (_, _, nx12, ny12, err12, mse12) = fit_line(lfps, sz, i1, i2);
                if mse12 > max_line_fit_mse {
                    continue;
                }

                let dot = ny01.mul_add(ny12, nx01 * nx12);
                if dot.abs() > max_dot {
                    continue;
                }

                for &i3 in maxima.iter().take(nmaxima).skip(m2 + 1) {
                    let (_, _, _, _, err23, mse23) = fit_line(lfps, sz, i2, i3);
                    if mse23 > max_line_fit_mse {
                        continue;
                    }

                    let (_, _, _, _, err30, mse30) = fit_line(lfps, sz, i3, i0);
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

    if best_error == f64::INFINITY || best_error / (sz as f64) >= max_line_fit_mse {
        return None;
    }

    Some(best_indices)
}

fn intersect_quad_lines(
    lfps: &[LineFitPt],
    sz: usize,
    best_indices: [usize; 4],
) -> Option<[[f32; 2]; 4]> {
    let mut lines = [(0.0, 0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        let i0 = best_indices[i];
        let i1 = best_indices[(i + 1) & 3];
        let (ex, ey, nx, ny, _, _) = fit_line(lfps, sz, i0, i1);
        lines[i] = (ex, ey, nx, ny);
    }

    let mut exact_corners = [[0.0f32; 2]; 4];
    for i in 0..4 {
        let (p0_x, p0_y, n0_x, n0_y) = lines[i];
        let (p1_x, p1_y, n1_x, n1_y) = lines[(i + 1) & 3];

        let a00 = n0_y;
        let a01 = -n1_y;
        let a10 = -n0_x;
        let a11 = n1_x;

        let b0 = -p0_x + p1_x;
        let b1 = -p0_y + p1_y;

        let det = a10.mul_add(-a01, a00 * a11);
        if det.abs() < 0.001 {
            return None;
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;
        let l0 = w01.mul_add(b1, w00 * b0);

        exact_corners[i] = [l0.mul_add(a00, p0_x) as f32, l0.mul_add(a10, p0_y) as f32];
    }

    Some(exact_corners)
}

/// Precomputes cumulative moments. Equivalent to `compute_lfps` in C.
/// Vectorization-friendly moment computation
fn compute_lfps(pts: &[FitPoint]) -> Vec<LineFitPt> {
    let len = pts.len();
    let mut lfps = Vec::with_capacity(len);

    let mut w_x = vec![0.0; len];
    let mut w_y = vec![0.0; len];
    let mut w_xx = vec![0.0; len];
    let mut w_yy = vec![0.0; len];
    let mut w_xy = vec![0.0; len];
    let mut weights = vec![0.0; len];

    for i in 0..len {
        let p = &pts[i];
        let x = p.x.mul_add(0.5, 0.5);
        let y = p.y.mul_add(0.5, 0.5);
        let w = (p.gx * p.gx + p.gy * p.gy).sqrt() + 1.0;

        weights[i] = w;
        w_x[i] = w * x;
        w_y[i] = w * y;
        w_xx[i] = w * x * x;
        w_yy[i] = w * y * y;
        w_xy[i] = w * x * y;
    }

    let mut sum_mx = 0.0;
    let mut sum_my = 0.0;
    let mut sum_mxx = 0.0;
    let mut sum_myy = 0.0;
    let mut sum_mxy = 0.0;
    let mut sum_w = 0.0;

    for i in 0..len {
        sum_mx += w_x[i];
        sum_my += w_y[i];
        sum_mxx += w_xx[i];
        sum_myy += w_yy[i];
        sum_mxy += w_xy[i];
        sum_w += weights[i];

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
fn fit_line(lfps: &[LineFitPt], sz: usize, i0: usize, i1: usize) -> (f64, f64, f64, f64, f64, f64) {
    let (mx, my, mxx, myy, mxy, w);
    let n: f64;

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

    let ex = mx / w;
    let ey = my / w;
    let cxx = ex.mul_add(-ex, mxx / w);
    let cxy = ex.mul_add(-ey, mxy / w);
    let cyy = ey.mul_add(-ey, myy / w);

    let eig_small = 0.5 * (cxx + cyy - (4.0 * cxy).mul_add(cxy, (cxx - cyy) * (cxx - cyy)).sqrt());
    let eig_large = 0.5 * (cxx + cyy + (4.0 * cxy).mul_add(cxy, (cxx - cyy) * (cxx - cyy)).sqrt());

    let nx1 = cxx - eig_large;
    let ny1 = cxy;
    let m1 = ny1.mul_add(ny1, nx1 * nx1);

    let nx2 = cxy;
    let ny2 = cyy - eig_large;
    let m2 = nx2.mul_add(nx2, ny2 * ny2);

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

/// A lightweight bilinear sampler for edge refinement
#[inline(always)]
fn sample_pixel_bilinear(image: &Image, x: f32, y: f32) -> Option<f32> {
    let x0 = x.trunc() as i32;
    let y0 = y.trunc() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    if x0 < 0 || y0 < 0 || x1 >= image.width as i32 || y1 >= image.height as i32 {
        return None;
    }

    let a = x.fract();
    let b = y.fract();

    let r0 = image.row(y0 as usize);
    let r1 = image.row(y1 as usize);

    let v00 = f32::from(r0[x0 as usize]);
    let v10 = f32::from(r0[x1 as usize]);
    let v01 = f32::from(r1[x0 as usize]);
    let v11 = f32::from(r1[x1 as usize]);

    Some((a * b).mul_add(
        v11,
        ((1.0 - a) * b).mul_add(
            v01,
            (a * (1.0 - b)).mul_add(v10, (1.0 - a) * (1.0 - b) * v00),
        ),
    ))
}

/// Refines the edges of a quadrilateral by sampling local image gradients.
/// Adjusts the `corners` in-place.
pub fn refine_edges(image: &Image, corners: &mut [[f32; 2]; 4]) {
    let mut lines = [[0.0f32; 4]; 4];

    for (edge, line) in lines.iter_mut().enumerate() {
        let a = edge;
        let b = (edge + 1) & 3;

        let mut nx = corners[b][1] - corners[a][1];
        let mut ny = -corners[b][0] + corners[a][0];

        let mag = (nx * nx + ny * ny).sqrt();
        if mag < 1e-5 {
            continue;
        }
        nx /= mag;
        ny /= mag;

        let nsamples = 16.max((mag / 8.0) as usize);

        let mut mx = 0.0;
        let mut my = 0.0;
        let mut mxx = 0.0;
        let mut mxy = 0.0;
        let mut myy = 0.0;
        let mut n = 0.0;

        for s in 0..nsamples {
            let alpha = (1 + s) as f32 / (nsamples + 1) as f32;
            let x0 = (1.0 - alpha).mul_add(corners[b][0], alpha * corners[a][0]);
            let y0 = (1.0 - alpha).mul_add(corners[b][1], alpha * corners[a][1]);

            let mut mn = 0.0;
            let mut m_count = 0.0;

            for step in 0..MAX_STEPS {
                let dist = STEP_LENGTH.mul_add(step as f32, -RANGE);

                let x1 = (dist + GRANGE).mul_add(nx, x0) - DELTA;
                let y1 = (dist + GRANGE).mul_add(ny, y0) - DELTA;

                let x2 = (dist - GRANGE).mul_add(nx, x0) - DELTA;
                let y2 = (dist - GRANGE).mul_add(ny, y0) - DELTA;

                if let (Some(g1), Some(g2)) = (
                    sample_pixel_bilinear(image, x1, y1),
                    sample_pixel_bilinear(image, x2, y2),
                ) {
                    if g1 < g2 {
                        continue;
                    }

                    let weight = (g2 - g1).powi(2);

                    mn += weight * dist;
                    m_count += weight;
                }
            }

            if m_count > 0.0 {
                let n0 = mn / m_count;
                let best_x = x0 + n0 * nx;
                let best_y = y0 + n0 * ny;

                mx += best_x;
                my += best_y;
                mxx += best_x * best_x;
                mxy += best_x * best_y;
                myy += best_y * best_y;
                n += 1.0;
            }
        }

        if n < 2.0 {
            *line = [
                f32::midpoint(corners[a][0], corners[b][0]),
                f32::midpoint(corners[a][1], corners[b][1]),
                nx,
                ny,
            ];
            continue;
        }

        let ex = mx / n;
        let ey = my / n;
        let cxx = mxx / n - ex * ex;
        let cxy = mxy / n - ex * ey;
        let cyy = myy / n - ey * ey;

        let normal_theta = 0.5 * (-2.0 * cxy).atan2(cyy - cxx);

        *line = [ex, ey, normal_theta.cos(), normal_theta.sin()];
    }

    for i in 0..4 {
        let a00 = lines[i][3];
        let a01 = -lines[(i + 1) & 3][3];
        let a10 = -lines[i][2];
        let a11 = lines[(i + 1) & 3][2];
        let b0 = -lines[i][0] + lines[(i + 1) & 3][0];
        let b1 = -lines[i][1] + lines[(i + 1) & 3][1];

        let det = a10.mul_add(-a01, a00 * a11);

        if det.abs() > 0.001 {
            let w00 = a11 / det;
            let w01 = -a01 / det;
            let l0 = w01.mul_add(b1, w00 * b0);

            corners[(i + 1) & 3][0] = l0.mul_add(a00, lines[i][0]);
            corners[(i + 1) & 3][1] = l0.mul_add(a10, lines[i][1]);
        }
    }
}
