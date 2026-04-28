use crate::apriltag::decode::Homography;
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub tag_size_mm: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Pose {
    pub r: Matrix3<f32>,
    pub t: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub distance_mm: f32,
    pub object_space_error: f32,
}

impl Pose {
    /// Extracts the Tait-Bryan Euler Angles and Distance from the R/t matrices.
    pub fn update_euler(&mut self) {
        self.pitch = (-self.r[(2, 0)]).asin().to_degrees();
        self.roll = self.r[(2, 1)].atan2(self.r[(2, 2)]).to_degrees();
        self.yaw = self.r[(1, 0)].atan2(self.r[(0, 0)]).to_degrees();
        self.distance_mm = self.t.norm();
    }
}

struct Alignment {
    r_t: Matrix3<f32>,
    r_z: Matrix3<f32>,
    r_gamma: Matrix3<f32>,
    t_initial: f32,
}

struct TransformedData {
    p_trans: [Vector3<f32>; 4],
    f_trans: [Matrix3<f32>; 4],
    avg_f_trans: Matrix3<f32>,
}

struct ErrorCoeffs {
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
}

/// Evaluates polynomial p at x. (p[0] + p[1]*x + p[2]*x^2 ...)
fn polyval(p: &[f32], x: f32) -> f32 {
    let mut ret = 0.0;
    let mut x_pow = 1.0;
    for &coeff in p {
        ret = coeff.mul_add(x_pow, ret);
        x_pow *= x;
    }
    ret
}

/// Numerically solve small degree polynomials using Newton's Method + Bisection.
fn solve_poly_approx(p: &[f32], roots: &mut Vec<f32>) {
    const MAX_ROOT: f32 = 1000.0;
    let degree = p.len() - 1;

    if degree == 1 {
        if p[0].abs() <= MAX_ROOT * p[1].abs() {
            roots.push(-p[0] / p[1]);
        }
        return;
    }

    let mut p_der = Vec::with_capacity(degree);
    for i in 0..degree {
        p_der.push((i + 1) as f32 * p[i + 1]);
    }

    let mut der_roots = Vec::new();
    solve_poly_approx(&p_der, &mut der_roots);

    for i in 0..=der_roots.len() {
        let min = if i == 0 { -MAX_ROOT } else { der_roots[i - 1] };
        let max = if i == der_roots.len() {
            MAX_ROOT
        } else {
            der_roots[i]
        };

        let val_min = polyval(p, min);
        let val_max = polyval(p, max);

        if val_min * val_max < 0.0 {
            let (mut lower, mut upper) = if val_min < val_max {
                (min, max)
            } else {
                (max, min)
            };
            let mut root = 0.5 * (lower + upper);
            let mut dx_old = upper - lower;
            let mut dx = dx_old;
            let mut f = polyval(p, root);
            let mut df = polyval(&p_der, root);

            for _ in 0..100 {
                if ((f + df * (upper - root)) * (f + df * (lower - root)) > 0.0)
                    || (2.0 * f).abs() > (dx_old * df).abs()
                {
                    dx_old = dx;
                    dx = 0.5 * (upper - lower);
                    root = lower + dx;
                } else {
                    dx_old = dx;
                    dx = -f / df;
                    root += dx;
                }

                if root == upper || root == lower {
                    break;
                }

                f = polyval(p, root);
                df = polyval(&p_der, root);

                if f > 0.0 {
                    upper = root;
                } else {
                    lower = root;
                }
            }
            roots.push(root);
        } else if val_max == 0.0 {
            roots.push(max);
        }
    }
}

/// Calculate the Projection Operator matrix F from an image point vector
#[allow(clippy::inline_always)]
#[inline(always)]
fn calculate_f(v: &Vector3<f32>) -> Matrix3<f32> {
    let outer = v * v.transpose();
    let inner = v.dot(v);
    outer / inner
}

/// Computes the initial guess using Homography Decomposition
fn initial_homography_guess(
    homo: &Homography,
    intrinsics: &CameraIntrinsics,
) -> (Matrix3<f32>, Vector3<f32>) {
    let scale = intrinsics.tag_size_mm / 2.0;

    let mut k_inv = Matrix3::<f32>::zeros();
    k_inv[(0, 0)] = 1.0 / intrinsics.fx;
    k_inv[(0, 2)] = -intrinsics.cx / intrinsics.fx;
    k_inv[(1, 1)] = 1.0 / intrinsics.fy;
    k_inv[(1, 2)] = -intrinsics.cy / intrinsics.fy;
    k_inv[(2, 2)] = 1.0;

    let h_prime = k_inv * homo.h;
    let h1 = h_prime.column(0);
    let h2 = h_prime.column(1);
    let h3 = h_prime.column(2);

    let lambda = 2.0 / (h1.norm() + h2.norm());

    let r1 = h1 * lambda;
    let r2 = h2 * lambda;
    let r3 = r1.cross(&r2);

    let r = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = r.svd(true, true);
    let u = svd.u.unwrap_or_else(Matrix3::identity);
    let v_t = svd.v_t.unwrap_or_else(Matrix3::identity);

    let mut r_ortho = u * v_t;
    if r_ortho.determinant() < 0.0 {
        let mut u_fixed = u;
        u_fixed.column_mut(2).scale_mut(-1.0);
        r_ortho = u_fixed * v_t;
    }

    let t = h3 * lambda * scale;
    (r_ortho, t)
}

/// Implementation of Orthogonal Iteration from Lu, 2000. Solves the 3D pose by iteratively minimizing object-space error.
pub fn orthogonal_iteration(
    v: &[Vector3<f32>; 4],
    p: &[Vector3<f32>; 4],
    mut r: Matrix3<f32>,
    mut t: Vector3<f32>,
    n_steps: usize,
) -> (Matrix3<f32>, Vector3<f32>, f32) {
    let mut p_mean = Vector3::zeros();
    for pi in p {
        p_mean += pi;
    }
    p_mean /= 4.0;

    let p_res = [p[0] - p_mean, p[1] - p_mean, p[2] - p_mean, p[3] - p_mean];

    let mut f_matrices = [Matrix3::zeros(); 4];
    let mut avg_f = Matrix3::zeros();
    for i in 0..4 {
        f_matrices[i] = calculate_f(&v[i]);
        avg_f += f_matrices[i];
    }
    avg_f /= 4.0;

    let i3 = Matrix3::identity();
    let m1 = i3 - avg_f;
    let m1_inv = m1.try_inverse().unwrap_or(i3);

    let mut prev_error = f32::MAX;

    for _ in 0..n_steps {
        let mut m2 = Vector3::zeros();
        for j in 0..4 {
            m2 += (f_matrices[j] - i3) * r * p[j];
        }
        m2 /= 4.0;
        t = m1_inv * m2;

        let mut q = [Vector3::zeros(); 4];
        let mut q_mean = Vector3::zeros();
        for j in 0..4 {
            q[j] = f_matrices[j] * (r * p[j] + t);
            q_mean += q[j];
        }
        q_mean /= 4.0;

        let mut m3 = Matrix3::zeros();
        for j in 0..4 {
            m3 += (q[j] - q_mean) * p_res[j].transpose();
        }

        let svd = m3.svd(true, true);
        let u = svd.u.unwrap_or(i3);
        let v_t = svd.v_t.unwrap_or(i3);

        r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fixed = u;
            u_fixed.column_mut(2).scale_mut(-1.0);
            r = u_fixed * v_t;
        }

        let mut error = 0.0;
        for j in 0..4 {
            let err_vec = (i3 - f_matrices[j]) * (r * p[j] + t);
            error += err_vec.dot(&err_vec);
        }

        if (prev_error - error).abs() < 1e-5 {
            break;
        }
        prev_error = error;
    }

    (r, t, prev_error)
}

/// The main entry point to extract the Tag's 3D Pose
pub fn estimate_tag_pose(
    homo: &Homography,
    corners: &[[f32; 2]; 4],
    intrinsics: &CameraIntrinsics,
) -> Pose {
    let s = intrinsics.tag_size_mm / 2.0;

    let p = [
        Vector3::new(-s, s, 0.0),
        Vector3::new(s, s, 0.0),
        Vector3::new(s, -s, 0.0),
        Vector3::new(-s, -s, 0.0),
    ];

    let mut v = [Vector3::zeros(); 4];
    for i in 0..4 {
        v[i] = Vector3::new(
            (corners[i][0] - intrinsics.cx) / intrinsics.fx,
            (corners[i][1] - intrinsics.cy) / intrinsics.fy,
            1.0,
        );
    }

    let (initial_r, initial_t) = initial_homography_guess(homo, intrinsics);

    let (r1, t1, err1) = orthogonal_iteration(&v, &p, initial_r, initial_t, 50);

    let mut best_r = r1;
    let mut best_t = t1;
    let mut lowest_error = err1;

    if let Some(r2_guess) = fix_pose_ambiguities(&v, &p, &t1, &r1) {
        let (r2, t2, err2) = orthogonal_iteration(&v, &p, r2_guess, Vector3::zeros(), 50);

        if err2 < err1 {
            best_r = r2;
            best_t = t2;
            lowest_error = err2;
        }
    }

    let mut pose = Pose {
        r: best_r,
        t: best_t,
        yaw: 0.0,
        pitch: 0.0,
        roll: 0.0,
        distance_mm: 0.0,
        object_space_error: lowest_error,
    };
    pose.update_euler();

    pose
}

/// Given a local minima of the pose error, tries to find the second possible "flip" minima.
pub fn fix_pose_ambiguities(
    v: &[Vector3<f32>; 4],
    p: &[Vector3<f32>; 4],
    t: &Vector3<f32>,
    r: &Matrix3<f32>,
) -> Option<Matrix3<f32>> {
    let align = compute_alignment(t, r);
    let data = transform_data(v, p, &align.r_z, &align.r_t);
    let coeffs = compute_error_coeffs(&data, &align.r_gamma);
    let poly = derive_polynomial(&coeffs);

    let mut roots = Vec::with_capacity(4);
    solve_poly_approx(&poly, &mut roots);

    let alt_min_t = find_alternative_minimum(&roots, &coeffs, align.t_initial)?;

    Some(reconstruct_rotation(alt_min_t, &align))
}

/// Computes the reference frame rotations needed to align the z-axis and extract the initial beta angle.
fn compute_alignment(t: &Vector3<f32>, r: &Matrix3<f32>) -> Alignment {
    let r_t_3 = t.normalize();
    let e_x = Vector3::new(1.0, 0.0, 0.0);

    let r_t_1 = (e_x - r_t_3 * e_x.dot(&r_t_3)).normalize();
    let r_t_2 = r_t_3.cross(&r_t_1);
    let r_t = Matrix3::from_rows(&[r_t_1.transpose(), r_t_2.transpose(), r_t_3.transpose()]);

    let r_1_prime = r_t * r;
    let mut r31 = r_1_prime[(2, 0)];
    let mut r32 = r_1_prime[(2, 1)];
    let mut hypotenuse = r31.hypot(r32);

    if hypotenuse < 1e-10 {
        r31 = 1.0;
        r32 = 0.0;
        hypotenuse = 1.0;
    }

    let r_z = Matrix3::new(
        r31 / hypotenuse,
        -r32 / hypotenuse,
        0.0,
        r32 / hypotenuse,
        r31 / hypotenuse,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let r_trans = r_1_prime * r_z;
    let sin_gamma = -r_trans[(0, 1)];
    let cos_gamma = r_trans[(1, 1)];
    let r_gamma = Matrix3::new(
        cos_gamma, -sin_gamma, 0.0, sin_gamma, cos_gamma, 0.0, 0.0, 0.0, 1.0,
    );

    let sin_beta = -r_trans[(2, 0)];
    let cos_beta = r_trans[(2, 2)];
    let t_initial = sin_beta.atan2(cos_beta);

    Alignment {
        r_t,
        r_z,
        r_gamma,
        t_initial,
    }
}

/// Transforms the object points and image rays into the aligned coordinate space.
fn transform_data(
    v: &[Vector3<f32>; 4],
    p: &[Vector3<f32>; 4],
    r_z: &Matrix3<f32>,
    r_t: &Matrix3<f32>,
) -> TransformedData {
    let mut p_trans = [Vector3::zeros(); 4];
    let mut f_trans = [Matrix3::zeros(); 4];
    let mut avg_f_trans = Matrix3::zeros();

    for i in 0..4 {
        p_trans[i] = r_z.transpose() * p[i];
        let v_trans = r_t * v[i];
        f_trans[i] = calculate_f(&v_trans);
        avg_f_trans += f_trans[i];
    }
    avg_f_trans /= 4.0;

    TransformedData {
        p_trans,
        f_trans,
        avg_f_trans,
    }
}

/// Formulates the scalar coefficients for the objective function.
fn compute_error_coeffs(data: &TransformedData, r_gamma: &Matrix3<f32>) -> ErrorCoeffs {
    let i3 = Matrix3::identity();
    let g = (i3 - data.avg_f_trans).try_inverse().unwrap_or(i3) / 4.0;

    let m1 = Matrix3::new(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0);
    let m2 = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);

    let mut b0 = Vector3::zeros();
    let mut b1 = Vector3::zeros();
    let mut b2 = Vector3::zeros();

    for i in 0..4 {
        let diff = i3 - data.f_trans[i];
        b0 += diff * r_gamma * data.p_trans[i];
        b1 += diff * r_gamma * m1 * data.p_trans[i];
        b2 += diff * r_gamma * m2 * data.p_trans[i];
    }

    let b0_ = g * b0;
    let b1_ = g * b1;
    let b2_ = g * b2;

    let mut a0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut a4 = 0.0;

    for i in 0..4 {
        let diff = i3 - data.f_trans[i];
        let c0 = diff * (r_gamma * data.p_trans[i] + b0_);
        let c1 = diff * (r_gamma * m1 * data.p_trans[i] + b1_);
        let c2 = diff * (r_gamma * m2 * data.p_trans[i] + b2_);

        a0 += c0.dot(&c0);
        a1 = 2.0f32.mul_add(c0.dot(&c1), a1);
        a2 += 2.0f32.mul_add(c0.dot(&c2), c1.dot(&c1));
        a3 = 2.0f32.mul_add(c1.dot(&c2), a3);
        a4 += c2.dot(&c2);
    }

    ErrorCoeffs { a0, a1, a2, a3, a4 }
}

/// Derives the polynomial coefficients used to find the critical points.
fn derive_polynomial(c: &ErrorCoeffs) -> [f32; 5] {
    [
        c.a1,
        4.0f32.mul_add(-c.a0, 2.0 * c.a2),
        3.0f32.mul_add(-c.a1, 3.0 * c.a3),
        2.0f32.mul_add(-c.a2, 4.0 * c.a4),
        -c.a3,
    ]
}

/// Evaluates the roots to ensure they represent a valid secondary minimum that is distinct from the initial pose.
fn find_alternative_minimum(roots: &[f32], c: &ErrorCoeffs, t_initial: f32) -> Option<f32> {
    let mut minima = Vec::with_capacity(4);

    for &t1 in roots {
        let t2 = t1 * t1;
        let t3 = t1 * t2;
        let t4 = t1 * t3;
        let t5 = t1 * t4;

        let derivative2 = c.a3.mul_add(
            t5,
            3.0f32.mul_add(c.a2, -6.0 * c.a4).mul_add(
                t4,
                6.0f32.mul_add(c.a1, -8.0 * c.a3).mul_add(
                    t3,
                    10.0f32
                        .mul_add(c.a0, 8.0f32.mul_add(-c.a2, 6.0 * c.a4))
                        .mul_add(
                            t2,
                            6.0f32
                                .mul_add(-c.a1, 3.0 * c.a3)
                                .mul_add(t1, 2.0f32.mul_add(-c.a0, c.a2)),
                        ),
                ),
            ),
        );

        if derivative2 >= 0.0 {
            let t_cur = 2.0 * t1.atan();
            if (t_cur - t_initial).abs() > 0.1 {
                minima.push(t1);
            }
        }
    }

    if minima.len() == 1 {
        Some(minima[0])
    } else {
        None
    }
}

/// Re-assembles the final pose rotation matrix using the new minimum angle.
fn reconstruct_rotation(t_cur: f32, align: &Alignment) -> Matrix3<f32> {
    let m1 = Matrix3::new(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0);
    let m2 = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    let i3 = Matrix3::identity();

    let mut r_beta = (m2 * t_cur + m1) * t_cur + i3;
    r_beta /= t_cur.mul_add(t_cur, 1.0);

    align.r_t.transpose() * align.r_gamma * r_beta * align.r_z
}
