use std::ops::AddAssign;

use num::Num;

pub struct TfIdfOps{

}

pub trait TfIdfCalc<N> 
where 
    N: Num + AddAssign + AddAssign + Copy
{
    /// TFの計算を行う。
    fn tf_calc(token_count: u32, token_sum: u64) -> N;
    /// TF のVecを返す。
    fn tf_vector(&self) -> Vec<(&str, N)>;
    /// IDFの計算を行う。
    // fn idf_calc(&self, token: &str) -> N;

}