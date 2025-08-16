use num::Num;

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
        vec.zip(other)
            .map(|(a, b)| ((a as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u8>, other: impl Iterator<Item = u8>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        for (a, b) in vec.zip(other) {
            vec_dot += ((a as u32 * a as u32 + u32::MAX - 1) / u32::MAX) as f64;
            other_dot += ((b as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64;
            dot += ((a as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64;
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
        vec.zip(other)
            .map(|(a, b)| ((a as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        for (a, b) in vec.zip(other) {
            vec_dot += ((a as u32 * a as u32 + u32::MAX - 1) / u32::MAX) as f64;
            other_dot += ((b as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64;
            dot += ((a as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64;
        }
        let cos = dot / (vec_dot.sqrt() * other_dot.sqrt());
        cos
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
        vec.zip(other)
            .map(|(a, b)| ((a * b + u32::MAX - 1) / u32::MAX) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u32>, other: impl Iterator<Item = u32>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        for (a, b) in vec.zip(other) {
            vec_dot += ((a * a + u32::MAX - 1) / u32::MAX) as f64;
            other_dot += ((b * b + u32::MAX - 1) / u32::MAX) as f64;
            dot += ((a * b + u32::MAX - 1) / u32::MAX) as f64;
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
        vec.zip(other)
            .map(|(a, b)| (a * b) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        let mut vec_dot = 0_f64;
        let mut other_dot = 0_f64;
        let mut dot = 0_f64;
        for (a, b) in vec.zip(other) {
            vec_dot += (a * a) as f64;
            other_dot += (b * b) as f64;
            dot += (a * b) as f64;
        }
        dot / (vec_dot.sqrt() * other_dot.sqrt())
    }

    fn chebyshev_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs() as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    fn euclidean_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| {
                let diff = a - b;
                (diff * diff) as f64
            })
            .sum::<f64>()
            .sqrt()
    }

    fn manhattan_distance(vec: impl Iterator<Item = f32>, other: impl Iterator<Item = f32>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum()
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
