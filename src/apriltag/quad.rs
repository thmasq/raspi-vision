use crate::apriltag::unionfind::Cluster;

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
pub fn find_quad_corners(cluster: &Cluster) -> Option<[[f32; 2]; 4]> {
    let min_cluster_pixels = 24;
    if cluster.points.len() < min_cluster_pixels {
        return None;
    }

    let mut pts: Vec<FitPoint> = cluster
        .points
        .iter()
        .map(|p| FitPoint {
            x: p.x as f64,
            y: p.y as f64,
            gx: p.gx as f64,
            gy: p.gy as f64,
            slope: 0.0,
        })
        .collect();

    let mut xmin = pts[0].x;
    let mut xmax = pts[0].x;
    let mut ymin = pts[0].y;
    let mut ymax = pts[0].y;

    for p in &pts {
        if p.x < xmin {
            xmin = p.x;
        }
        if p.x > xmax {
            xmax = p.x;
        }
        if p.y < ymin {
            ymin = p.y;
        }
        if p.y > ymax {
            ymax = p.y;
        }
    }

    let cx = (xmin + xmax) * 0.5 + 0.05118;
    let cy = (ymin + ymax) * 0.5 - 0.028581;

    for p in pts.iter_mut() {
        let mut dx = p.x - cx;
        let mut dy = p.y - cy;

        let quadrant = if dy > 0.0 {
            if dx > 0.0 { 131072.0 } else { 65536.0 }
        } else {
            if dx > 0.0 { 0.0 } else { -131072.0 }
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
        p.slope = quadrant + (dy / dx);
    }

    pts.sort_unstable_by(|a, b| a.slope.partial_cmp(&b.slope).unwrap());

    let lfps = compute_lfps(&pts);

    let sz = pts.len();
    let ksz = 20.min(sz / 12);
    if ksz < 2 {
        return None;
    }

    let mut errs = vec![0.0; sz];
    for i in 0..sz {
        let (_, _, _, _, err, _) = fit_line(&lfps, sz, (i + sz - ksz) % sz, (i + ksz) % sz);
        errs[i] = err;
    }

    let sigma = 1.0_f64;
    let cutoff = 0.05_f64;
    let mut fsz = (-cutoff.ln() * 2.0 * sigma * sigma).sqrt() as i32 + 1;
    fsz = 2 * fsz + 1;

    let mut f = Vec::with_capacity(fsz as usize);
    for i in 0..fsz {
        let j = (i - fsz / 2) as f64;
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

        maxima = maxima
            .into_iter()
            .zip(maxima_errs.into_iter())
            .filter(|(_, err)| *err > threshold)
            .map(|(idx, _)| idx)
            .collect();
    }

    let nmaxima = maxima.len();
    let mut best_indices = [0; 4];
    let mut best_error = f64::INFINITY;
    let max_dot = (25.0_f64 * std::f64::consts::PI / 180.0).cos();
    let max_line_fit_mse = 10.0_f64;

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

    if best_error == f64::INFINITY || best_error / (sz as f64) >= max_line_fit_mse {
        return None;
    }

    let mut lines = [(0.0, 0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        let i0 = best_indices[i];
        let i1 = best_indices[(i + 1) & 3];
        let (ex, ey, nx, ny, _, _) = fit_line(&lfps, sz, i0, i1);
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

        let det = a00 * a11 - a10 * a01;
        if det.abs() < 0.001 {
            return None;
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;
        let l0 = w00 * b0 + w01 * b1;

        exact_corners[i] = [(p0_x + l0 * a00) as f32, (p0_y + l0 * a10) as f32];
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
        let x = p.x * 0.5 + 0.5;
        let y = p.y * 0.5 + 0.5;
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
