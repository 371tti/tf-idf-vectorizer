use num::{traits::MulAdd, Num};

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
    fn cosine_similarity(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
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

impl Compare<u8> for DefaultCompare {
    fn dot(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        // NOTE: 元は u32::MAX で割っていたため常に 0 になり比較不能だった。
        // u8 の量子化レンジに合わせ u8::MAX を使用。
        let max = u8::MAX as u32;
        vec.zip(other)
            .map(|(a, b)| (( (a as u32).mul_add(b as u32, max - 1) ) / max) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        let max = u8::MAX as u32;
        for (a, b) in vec.zip(other) {
            vec_dot += (( (a as u32).mul_add(a as u32, max - 1) ) / max) as f64;
            other_dot += (( (b as u32).mul_add(b as u32, max - 1) ) / max) as f64;
            dot += (( (a as u32).mul_add(b as u32, max - 1) ) / max) as f64;
        }
        dot / (vec_dot.sqrt() * other_dot.sqrt())
    }

    fn euclidean_distance(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    fn manhattan_distance(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .sum()
    }

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
    fn dot(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        // u16 でも正しいスケール (u16::MAX) を使用。
        let max = u16::MAX as u32;
        vec.zip(other)
            .map(|(a, b)| (( (a as u32).mul_add(b as u32, max - 1) ) / max) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        let max = u16::MAX as u32;
        for (a, b) in vec.zip(other) {
            vec_dot += (((a as u32).mul_add(a as u32, max - 1)) / max) as f64;
            other_dot += (((b as u32).mul_add(b as u32, max - 1)) / max) as f64;
            dot += (((a as u32).mul_add(b as u32, max - 1)) / max) as f64;
        }
        dot / (vec_dot.sqrt() * other_dot.sqrt())
    }

    /// ?
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
    fn manhattan_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i32 - b as i32;
                diff.abs() as f64
            })
            .sum()
    }

    /// ?
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

    fn cosine_similarity(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        let max = u32::MAX as u128;
        for (a, b) in vec.zip(other) {
            let aa = (a as u128).mul_add(a as u128, max - 1);
            let bb = (b as u128).mul_add(b as u128, max - 1);
            let ab = (a as u128).mul_add(b as u128, max - 1);
            vec_dot += (aa / max) as f64;
            other_dot += (bb / max) as f64;
            dot += (ab / max) as f64;
        }
        dot / (vec_dot.sqrt() * other_dot.sqrt())
    }

    fn euclidean_distance(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i64 - b as i64;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    fn manhattan_distance(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a as i64 - b as i64;
                diff.abs() as f64
            })
            .sum()
    }

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
    fn dot(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            acc += a * b; // f32 のまま蓄積
        }
        acc as f64
    }

    fn cosine_similarity(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut sum_a2: f32 = 0.0;
        let mut sum_b2: f32 = 0.0;
        let mut sum_ab: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            sum_a2 += a * a;
            sum_b2 += b * b;
            sum_ab += a * b;
        }
        if sum_a2 == 0.0 || sum_b2 == 0.0 { return 0.0; }
        (sum_ab as f64) / ((sum_a2.sqrt() * sum_b2.sqrt()) as f64)
    }

    fn chebyshev_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut maxd: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            let d = (a - b).abs();
            if d > maxd { maxd = d; }
        }
        maxd as f64
    }

    fn euclidean_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            let d = a - b;
            acc += d * d;
        }
        (acc.sqrt()) as f64
    }

    fn manhattan_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut acc: f32 = 0.0;
        for (a, b) in vec.zip(other) {
            acc += (a - b).abs();
        }
        acc as f64
    }
}

impl Compare<f64> for DefaultCompare {
    fn dot(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| a * b)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        for (a, b) in vec.zip(other) {
            vec_dot += a * a;
            other_dot += b * b;
            dot += a * b;
        }
        dot / (vec_dot.sqrt() * other_dot.sqrt())
    }

    fn euclidean_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }

    fn manhattan_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    fn chebyshev_distance(vec: impl Iterator<Item = f64>, other: impl Iterator<Item = f64>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}
