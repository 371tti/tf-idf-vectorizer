use std::{collections::HashMap, ops::{AddAssign, MulAssign}};

use num::Num;

use crate::{utils::math::vector::{math::VecOps, ZeroSpVecTrait}, ZeroSpVec};

pub trait TfIdfCalc<N>: 
where 
    N: Num + AddAssign + AddAssign + Copy + MulAssign
{
    /// TFの計算を行う。
    fn tf_calc(token_count: u32, token_sum: u64) -> N;
    /// TF のVecを返す。
    fn idf_calc(token_count: u64, doc_sum: u64) -> N;
    /// TF ZeroSpVecを返す。
    fn tf_vec(
        token_dim_map: &HashMap<String, usize>,
        token_vec: &Vec<(String, u64)>,
        token_sum: u64) -> impl VecOps<N> {
            let mut zero_sp_vec = ZeroSpVec::with_capacity(token_vec.len());
            for (token, count) in token_vec {
                if let Some(&index) = token_dim_map.get(token) {
                    let tf = Self::tf_calc(*count as u32, token_sum);
                    zero_sp_vec.raw_push(index, tf);
                }
            }
            zero_sp_vec
        }
    /// IDF ZeroSpVecを返す。
    fn idf_vec(token_count: u64, doc_sum: u64) -> impl VecOps<N>;
}