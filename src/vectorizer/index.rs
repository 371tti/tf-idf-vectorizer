use std::ops::{AddAssign, MulAssign};

use num::Num;

use crate::utils::{math::vector::ZeroSpVec, normalizer::NormalizedMultiply};

pub struct Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply{
    pub matrix: Vec<ZeroSpVec<N>>,
}