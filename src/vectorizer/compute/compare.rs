use num::{traits::MulAdd, Num};
use std::cmp::Ordering;

pub trait Compare<N>
where
    N: Num + Copy,
{
    /// dot積 
    /// d(a, b) = Σ(a_i * b_i)
    fn dot(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    /// コサイン類似度
    /// cos(θ) = Σ(a_i * b_i) / (||a|| * ||b||)
    /// ||a|| = sqrt(Σ(a_i^2))
    fn cosine_similarity(vec: impl Iterator<Item = (usize, N)>, other: impl Iterator<Item = (usize, N)>) -> f64;
    /// ユークリッド距離
    /// d(a, b) = sqrt(Σ((a_i - b_i)^2))
    fn euclidean_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    /// マンハッタン距離
    /// d(a, b) = Σ(|a_i - b_i|)
    fn manhattan_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    /// チェビシェフ距離
    /// d(a, b) = max(|a_i - b_i|)
    fn chebyshev_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
}

#[derive(Debug)]
pub struct DefaultCompare;

/// impl Compare for u8, u16, u32, f32, f64
impl Compare<u8> for DefaultCompare {
    #[inline(always)]
    fn dot(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        // NOTE: 元は u32::MAX で割っていたため常に 0 になり比較不能だった。
        // u8 の量子化レンジに合わせ u8::MAX を使用。
        let max = u8::MAX as u32;
        // Normalize dot product to [0, 1] range for quantized u8 values.
        vec.zip(other)
            .map(|(a, b)| ((a as u32 * b as u32) as f64 / max as f64))
            .sum()
    }

    #[inline(always)]
    fn cosine_similarity(vec: impl Iterator<Item = (usize, u8)>, other: impl Iterator<Item = (usize, u8)>) -> f64 {
        let max = u8::MAX as u32;
        let mut a_it = vec.fuse();
        let mut b_it = other.fuse();
        let mut a_next = a_it.next();
        let mut b_next = b_it.next();
        let mut norm_a = 0_f64;
        let mut norm_b = 0_f64;
        let mut dot = 0_f64;
        while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
            match ia.cmp(&ib) {
                Ordering::Equal => {
                    norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64;
                    norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64;
                    dot += (((va as u32).mul_add(vb as u32, max - 1)) / max) as f64;
                    a_next = a_it.next();
                    b_next = b_it.next();
                }
                Ordering::Less => {
                    norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64;
                    a_next = a_it.next();
                }
                Ordering::Greater => {
                    norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64;
                    b_next = b_it.next();
                }
            }
        }
        while let Some((_, va)) = a_next { norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64; a_next = a_it.next(); }
        while let Some((_, vb)) = b_next { norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64; b_next = b_it.next(); }
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a.sqrt() * norm_b.sqrt()) }
    }

    #[inline(always)]
    fn euclidean_distance(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    #[inline(always)]
    fn manhattan_distance(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .sum()
    }

    #[inline(always)]
    fn chebyshev_distance(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

impl Compare<u16> for DefaultCompare {
    #[inline(always)]
    fn dot(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        // u16 でも正しいスケール (u16::MAX) を使用。
        let max = u16::MAX as u32;
        vec.zip(other)
            .map(|(a, b)| (( (a as u32).mul_add(b as u32, max - 1) ) / max) as f64)
            .sum()
    }

    #[inline(always)]
    fn cosine_similarity(vec: impl Iterator<Item = (usize, u16)>, other: impl Iterator<Item = (usize, u16)>) -> f64 {
        let max = u16::MAX as u32;
        let mut a_it = vec.fuse();
        let mut b_it = other.fuse();
        let mut a_next = a_it.next();
        let mut b_next = b_it.next();
        let mut norm_a = 0_f64;
        let mut norm_b = 0_f64;
        let mut dot = 0_f64;
        while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
            match ia.cmp(&ib) {
                Ordering::Equal => {
                    norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64;
                    norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64;
                    dot += (((va as u32).mul_add(vb as u32, max - 1)) / max) as f64;
                    a_next = a_it.next();
                    b_next = b_it.next();
                }
                Ordering::Less => { norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64; a_next = a_it.next(); }
                Ordering::Greater => { norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64; b_next = b_it.next(); }
            }
        }
        while let Some((_, va)) = a_next { norm_a += (((va as u32).mul_add(va as u32, max - 1)) / max) as f64; a_next = a_it.next(); }
        while let Some((_, vb)) = b_next { norm_b += (((vb as u32).mul_add(vb as u32, max - 1)) / max) as f64; b_next = b_it.next(); }
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a.sqrt() * norm_b.sqrt()) }
    }

    /// ?
    #[inline(always)]
    fn euclidean_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    /// ?
    #[inline(always)]
    fn manhattan_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .sum()
    }

    /// ?
    #[inline(always)]
    fn chebyshev_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

impl Compare<u32> for DefaultCompare {
    #[inline(always)]
    fn dot(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        // u32 の a*b は 64bit を超える可能性があるので u128 で計算。
        let max = u32::MAX as u128;
        vec.zip(other)
            .map(|(a, b)| {
                let prod = (a as u128).mul_add(b as u128, max - 1);
                (prod / max) as f64
            })
            .sum()
    }

    #[inline(always)]
    fn cosine_similarity(vec: impl Iterator<Item = (usize, u32)>, other: impl Iterator<Item = (usize, u32)>) -> f64 {
        let max = u32::MAX as u128;
        let mut a_it = vec.fuse();
        let mut b_it = other.fuse();
        let mut a_next = a_it.next();
        let mut b_next = b_it.next();
        let mut norm_a = 0_f64;
        let mut norm_b = 0_f64;
        let mut dot = 0_f64;
        while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
            match ia.cmp(&ib) {
                Ordering::Equal => {
                    let aa = (va as u128).mul_add(va as u128, max - 1);
                    let bb = (vb as u128).mul_add(vb as u128, max - 1);
                    let ab = (va as u128).mul_add(vb as u128, max - 1);
                    norm_a += (aa / max) as f64;
                    norm_b += (bb / max) as f64;
                    dot += (ab / max) as f64;
                    a_next = a_it.next();
                    b_next = b_it.next();
                }
                Ordering::Less => {
                    let aa = (va as u128).mul_add(va as u128, max - 1);
                    norm_a += (aa / max) as f64;
                    a_next = a_it.next();
                }
                Ordering::Greater => {
                    let bb = (vb as u128).mul_add(vb as u128, max - 1);
                    norm_b += (bb / max) as f64;
                    b_next = b_it.next();
                }
            }
        }
        while let Some((_, va)) = a_next { let aa = (va as u128).mul_add(va as u128, max -1); norm_a += (aa / max) as f64; a_next = a_it.next(); }
        while let Some((_, vb)) = b_next { let bb = (vb as u128).mul_add(vb as u128, max -1); norm_b += (bb / max) as f64; b_next = b_it.next(); }
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a.sqrt() * norm_b.sqrt()) }
    }

    #[inline(always)]
    fn euclidean_distance(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i64 - b as i64;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    #[inline(always)]
    fn manhattan_distance(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i64 - b as i64;
                diff.abs() as f64
            })
            .sum()
    }

    #[inline(always)]
    fn chebyshev_distance(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i64 - b as i64;
                diff.abs() as f64
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

impl Compare<f32> for DefaultCompare {
    #[inline(always)]
    fn dot(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            acc += a * b; // f32 のまま蓄積
        }
        acc as f64
    }

    #[inline(always)]
    fn cosine_similarity(vec: impl Iterator<Item = (usize, f32)>, other: impl Iterator<Item = (usize, f32)>) -> f64 {
        let mut a_it = vec.fuse();
        let mut b_it = other.fuse();
        let mut a_next = a_it.next();
        let mut b_next = b_it.next();
        let mut sum_a2: f64 = 0.0;
        let mut sum_b2: f64 = 0.0;
        let mut sum_ab: f64 = 0.0;
        while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
            match ia.cmp(&ib) {
                Ordering::Equal => { sum_a2 += (va * va) as f64; sum_b2 += (vb * vb) as f64; sum_ab += (va * vb) as f64; a_next = a_it.next(); b_next = b_it.next(); }
                Ordering::Less => { sum_a2 += (va * va) as f64; a_next = a_it.next(); }
                Ordering::Greater => { sum_b2 += (vb * vb) as f64; b_next = b_it.next(); }
            }
        }
        while let Some((_, va)) = a_next { sum_a2 += (va * va) as f64; a_next = a_it.next(); }
        while let Some((_, vb)) = b_next { sum_b2 += (vb * vb) as f64; b_next = b_it.next(); }
        if sum_a2 == 0.0 || sum_b2 == 0.0 { 0.0 } else { sum_ab / (sum_a2.sqrt() * sum_b2.sqrt()) }
    }

    #[inline(always)]
    fn chebyshev_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut maxd: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            let d = (a - b).abs();
            if d > maxd { maxd = d; }
        }
        maxd as f64
    }

    #[inline(always)]
    fn euclidean_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            let d = a - b;
            acc += d * d;
        }
        (acc.sqrt()) as f64
    }

    #[inline(always)]
    fn manhattan_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            acc += (a - b).abs();
        }
        acc as f64
    }
}

impl Compare<f64> for DefaultCompare {
    #[inline(always)]
    fn dot(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| a * b)
            .sum()
    }

    #[inline(always)]
    fn cosine_similarity(vec: impl Iterator<Item = (usize, f64)>, other: impl Iterator<Item = (usize, f64)>) -> f64 {
        let mut a_it = vec.fuse();
        let mut b_it = other.fuse();
        let mut a_next = a_it.next();
        let mut b_next = b_it.next();
        let mut norm_a = 0_f64;
        let mut norm_b = 0_f64;
        let mut dot = 0_f64;
        while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
            match ia.cmp(&ib) {
                Ordering::Equal => { norm_a += va * va; norm_b += vb * vb; dot += va * vb; a_next = a_it.next(); b_next = b_it.next(); }
                Ordering::Less => { norm_a += va * va; a_next = a_it.next(); }
                Ordering::Greater => { norm_b += vb * vb; b_next = b_it.next(); }
            }
        }
        while let Some((_, va)) = a_next { norm_a += va * va; a_next = a_it.next(); }
        while let Some((_, vb)) = b_next { norm_b += vb * vb; b_next = b_it.next(); }
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a.sqrt() * norm_b.sqrt()) }
    }

    #[inline(always)]
    fn euclidean_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }

    #[inline(always)]
    fn manhattan_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    #[inline(always)]
    fn chebyshev_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}
