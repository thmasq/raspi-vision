use crate::apriltag::unionfind::{Blob, Segment};

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub slope: f32,
}

/// Extracts the boundary points from a blob's RLE segments and orders them
/// clockwise around the centroid of the blob.
pub fn extract_ordered_boundary(blob: &Blob, segments: &[Segment]) -> Vec<Point> {
    let expected_perimeter = (blob.pixel_count as f32).sqrt() as usize * 4;
    let mut boundary = Vec::with_capacity(expected_perimeter);

    let mut sum_x: u32 = 0;
    let mut sum_y: u32 = 0;

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
        sum_x += seg.x_start as u32;
        sum_y += seg.y as u32;

        if seg.x_start != seg.x_end {
            boundary.push(right);
            sum_x += seg.x_end as u32;
            sum_y += seg.y as u32;
        }
    }

    let n = boundary.len() as f32;
    if n < 8.0 {
        return boundary;
    }

    let cx = (sum_x as f32) / n;
    let cy = (sum_y as f32) / n;

    for pt in boundary.iter_mut() {
        let dx = pt.x - cx;
        let dy = pt.y - cy;

        let quadrant = if dy > 0.0 {
            if dx > 0.0 { 0.0 } else { 1.0 }
        } else {
            if dx < 0.0 { 2.0 } else { 3.0 }
        };

        let slope = if dx.abs() > 1e-5 {
            dy / dx
        } else {
            9999.0 * dy.signum()
        };
        pt.slope = quadrant * 1000.0 + slope;
    }

    // sort_unstable_by is highly optimized in the standard library
    boundary.sort_unstable_by(|a, b| a.slope.partial_cmp(&b.slope).unwrap());

    boundary
}

/// Fits a 2D line to a subset of boundary points.
/// Returns: (MeanX, MeanY, NormalX, NormalY, MeanSquaredError)
pub fn fit_line(points: &[Point]) -> Option<(f32, f32, f32, f32, f32)> {
    let n = points.len() as f32;
    if n < 2.0 {
        return None;
    }

    // 1. Calculate Mean
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for p in points {
        sum_x += p.x;
        sum_y += p.y;
    }
    let ex = sum_x / n;
    let ey = sum_y / n;

    // 2. Calculate Covariance Matrix Elements
    // LLVM will automatically vectorize this loop using ARM NEON
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for p in points {
        let dx = p.x - ex;
        let dy = p.y - ey;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= n;
    cyy /= n;
    cxy /= n;

    // 3. Analytic Eigendecomposition of the 2x2 Covariance Matrix
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;

    let discriminant = ((trace * trace) / 4.0 - det).abs().sqrt();
    let eig_small = (trace / 2.0) - discriminant;
    let eig_large = (trace / 2.0) + discriminant;

    let mut nx = cxx - eig_large;
    let mut ny = cxy;
    let length = (nx * nx + ny * ny).sqrt();

    if length > 1e-6 {
        nx /= length;
        ny /= length;
    } else {
        nx = 0.0;
        ny = 0.0;
    }

    Some((ex, ey, nx, ny, eig_small))
}

/// Finds the 4 mathematical sub-pixel corners of an AprilTag quad
/// from an ordered boundary of points.
pub fn find_quad_corners(boundary: &[Point]) -> Option<[Point; 4]> {
    let sz = boundary.len();
    if sz < 24 {
        return None;
    }

    // The kernel size dictates how many points on either side of the target are used to fit the line.
    let ksz = 20.min(sz / 12);
    if ksz < 2 {
        return None;
    }

    let mut errs = Vec::with_capacity(sz);
    let mut window_buf = Vec::with_capacity(ksz * 2 + 1);

    for i in 0..sz {
        window_buf.clear();

        for k in 0..=(ksz * 2) {
            let idx = (i + sz + k - ksz) % sz;
            window_buf.push(boundary[idx]);
        }

        if let Some((_, _, _, _, mse)) = fit_line(&window_buf) {
            errs.push(mse);
        } else {
            errs.push(0.0);
        }
    }

    let mut smoothed_errs = Vec::with_capacity(sz);
    for i in 0..sz {
        let prev2 = errs[(i + sz - 2) % sz];
        let prev1 = errs[(i + sz - 1) % sz];
        let curr = errs[i];
        let next1 = errs[(i + 1) % sz];
        let next2 = errs[(i + 2) % sz];

        // 1-4-6-4-1 kernel approximation
        let smoothed = (prev2 + 4.0 * prev1 + 6.0 * curr + 4.0 * next1 + next2) / 16.0;
        smoothed_errs.push(smoothed);
    }

    let mut maxima = Vec::with_capacity(16);
    for i in 0..sz {
        let prev = smoothed_errs[(i + sz - 1) % sz];
        let curr = smoothed_errs[i];
        let next = smoothed_errs[(i + 1) % sz];

        if curr > prev && curr > next {
            maxima.push(i);
        }
    }

    if maxima.len() < 4 {
        return None;
    }

    if maxima.len() > 4 {
        maxima.sort_unstable_by(|&a, &b| smoothed_errs[b].partial_cmp(&smoothed_errs[a]).unwrap());
        maxima.truncate(4);
    }

    maxima.sort_unstable();

    // 1. Fit 4 continuous lines using the discrete points between our 4 maxima
    let mut lines = [(0.0, 0.0, 0.0, 0.0); 4]; // (Ex, Ey, nx, ny)
    for i in 0..4 {
        let start_idx = maxima[i];
        let end_idx = maxima[(i + 1) % 4];

        let mut edge_points = Vec::new();
        let mut curr = start_idx;
        loop {
            edge_points.push(boundary[curr]);
            if curr == end_idx {
                break;
            }
            curr = (curr + 1) % sz;
        }

        if let Some((ex, ey, nx, ny, _)) = fit_line(&edge_points) {
            lines[i] = (ex, ey, nx, ny);
        } else {
            return None;
        }
    }

    // 2. Intersect the 4 lines to calculate the exact sub-pixel corners
    let mut exact_corners = [Point {
        x: 0.0,
        y: 0.0,
        slope: 0.0,
    }; 4];

    for i in 0..4 {
        let (p0_x, p0_y, n0_x, n0_y) = lines[i];
        let (p1_x, p1_y, n1_x, n1_y) = lines[(i + 1) % 4];

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
            x: p0_x + l0 * a00,
            y: p0_y + l0 * a10,
            slope: 0.0,
        };
    }

    Some(exact_corners)
}
