use num::Num;

pub trait Compare<N>
where
    N: Num + Copy,
{
    /// dot積
    fn dot(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    fn cosine_similarity(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    fn euclidean_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    fn manhattan_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
    fn chebyshev_distance(vec: impl Iterator<Item = N>, other: impl Iterator<Item = N>) -> f64;
}

pub struct DefaultCompare;

impl Compare<u16> for DefaultCompare {
    fn dot(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        vec.zip(other)
            .map(|(a, b)| ((a as u32 * b as u32 + u32::MAX - 1) / u32::MAX) as f64)
            .sum()
    }

    fn cosine_similarity(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        let vec_iter = vec.iter()
        let vec_norm = DefaultCompare::dot(vec, vec);
        let other_norm = DefaultCompare::dot(other, other);
        DefaultCompare::dot(vec, other) / (vec_norm * other_norm).sqrt()
    }

    fn euclidean_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        todo!()
    }

    fn manhattan_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        todo!()
    }

    fn chebyshev_distance(vec: impl Iterator<Item = u16>, other: impl Iterator<Item = u16>) -> f64 {
        todo!()
    }
}