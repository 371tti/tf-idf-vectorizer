use serde::{Deserialize, Deserializer, Serialize};
use std::ops::{AddAssign, MulAssign};
use num::Num;
use crate::utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}};
use crate::TokenFrequency;

use super::Index;

impl<'de, N> Deserialize<'de> for Index<N>
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded + Deserialize<'de>,
    f64: IntoNormalizer<N>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>
    {
        // ヘルパー構造体：doc_token_countはOption指定し、
        // なければNoneとなるようにする
        #[derive(Deserialize)]
        struct IndexHelper<N>
        where
            N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded,
        {
            matrix: Vec<ZeroSpVec<N>>,
            #[serde(default)]
            doc_token_count: Option<Vec<u64>>,
            doc_id: Vec<String>,
            corpus_token_freq: TokenFrequency,
        }

        let helper = IndexHelper::<N>::deserialize(deserializer)?;
        let doc_token_count = match helper.doc_token_count {
            Some(v) => v,
            None => vec![1u64; helper.matrix.len()],
        };
        Ok(Index {
            matrix: helper.matrix,
            doc_token_count,
            doc_id: helper.doc_id,
            corpus_token_freq: helper.corpus_token_freq,
        })
    }
}