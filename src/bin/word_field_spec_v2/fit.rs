pub fn tiles(dim: i64, tile_size: i64) -> i64 {
    dim / tile_size
}

pub fn bits_mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        ((1u128 << bits) - 1) as u64
    }
}

pub fn to_signed(v: u64, bits: u8) -> i64 {
    let mask = bits_mask(bits);
    let vv = v & mask;
    let sign = 1u64 << (bits - 1);
    if (vv & sign) != 0 {
        (vv as i128 - (1i128 << bits)) as i64
    } else {
        vv as i64
    }
}

pub fn from_signed(v: i64, bits: u8) -> u64 {
    let mask = bits_mask(bits) as i128;
    ((v as i128) & mask) as u64
}

pub fn interp_domain(
    low: u64,
    high: u64,
    xl: f64,
    xh: f64,
    xt: f64,
    bits: u8,
    domain: &str,
) -> i128 {
    if (xh - xl).abs() <= f64::EPSILON {
        return low as i128;
    }

    let frac = (xt - xl) / (xh - xl);
    match domain {
        "u" => {
            let y = low as f64 + ((high as f64 - low as f64) * frac);
            y.round() as i128
        }
        "s" => {
            let l = to_signed(low, bits);
            let h = to_signed(high, bits);
            let y = l as f64 + ((h as f64 - l as f64) * frac);
            from_signed(y.round() as i64, bits) as i128
        }
        "mod" => {
            let ring = 1i128 << bits;
            let half = ring / 2;
            let low_i = (low as i128).rem_euclid(ring);
            let high_i = (high as i128).rem_euclid(ring);
            let delta = (high_i - low_i + half).rem_euclid(ring) - half;
            let y = (low_i as f64 + (delta as f64 * frac)).round() as i128;
            y.rem_euclid(ring)
        }
        _ => {
            let y = low as f64 + ((high as f64 - low as f64) * frac);
            y.round() as i128
        }
    }
}

pub fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut m = [[0.0f64; 4]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = a[r][c];
        }
        m[r][3] = b[r];
    }

    for col in 0..3 {
        let mut pivot = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() <= 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
        }

        let p = m[col][col];
        for j in col..4 {
            m[col][j] /= p;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() <= 1e-12 {
                continue;
            }
            for j in col..4 {
                m[r][j] -= factor * m[col][j];
            }
        }
    }

    Some([m[0][3], m[1][3], m[2][3]])
}

pub fn fit_three_point_predict(
    lo_x: f64,
    lo_y: f64,
    mid_x: f64,
    mid_y: f64,
    hi_x: f64,
    hi_y: f64,
    target_x: f64,
) -> i128 {
    if (lo_y - mid_y).abs() <= 1e-12 && (mid_y - hi_y).abs() <= 1e-12 {
        return lo_y.round() as i128;
    }

    if (hi_x - lo_x).abs() > 1e-12 {
        let mid_lin = lo_y + ((hi_y - lo_y) * ((mid_x - lo_x) / (hi_x - lo_x)));
        if (mid_lin - mid_y).abs() <= 1e-9 {
            let yt = lo_y + ((hi_y - lo_y) * ((target_x - lo_x) / (hi_x - lo_x)));
            return yt.round() as i128;
        }
    }

    let mat = [
        [lo_x * lo_x, lo_x, 1.0],
        [mid_x * mid_x, mid_x, 1.0],
        [hi_x * hi_x, hi_x, 1.0],
    ];
    if let Some(sol) = solve_3x3(mat, [lo_y, mid_y, hi_y]) {
        return (sol[0] * target_x * target_x + sol[1] * target_x + sol[2]).round() as i128;
    }

    if (hi_x - lo_x).abs() > 1e-12 {
        let yt = lo_y + ((hi_y - lo_y) * ((target_x - lo_x) / (hi_x - lo_x)));
        yt.round() as i128
    } else {
        lo_y.round() as i128
    }
}

pub fn decode_domain_value(v: u64, bits: u8, domain: &str) -> f64 {
    match domain {
        "s" => to_signed(v, bits) as f64,
        _ => v as f64,
    }
}

pub fn encode_domain_value(v: i128, bits: u8, domain: &str) -> i128 {
    match domain {
        "s" => from_signed(v as i64, bits) as i128,
        "mod" => {
            let ring = 1i128 << bits;
            v.rem_euclid(ring)
        }
        _ => v,
    }
}
