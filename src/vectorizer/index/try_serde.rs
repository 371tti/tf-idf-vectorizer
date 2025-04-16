use num::Num;
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use serde::de::{self, Visitor, MapAccess};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign};

use crate::utils::normalizer::{NormalizedBounded, NormalizedMultiply};
use crate::{TokenFrequency, ZeroSpVec};

use super::Index;

/// ヘルパー用のenum。通常は (ZeroSpVec<N>, u64) のタプル形式、
/// もしくは ZeroSpVec<N> のみの場合は u64 = 1 として扱う
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum MatrixEntry<N>
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded,
{
    Full((ZeroSpVec<N>, u64)),
    OnlyVec(ZeroSpVec<N>),
}

impl<N> From<MatrixEntry<N>> for (ZeroSpVec<N>, u64)
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded,
{
    fn from(entry: MatrixEntry<N>) -> Self {
        match entry {
            MatrixEntry::Full(pair) => pair,
            MatrixEntry::OnlyVec(vec) => (vec, 1),
        }
    }
}

impl<N> Serialize for Index<N>
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        // matrixフィールドは MatrixEntry に変換してシリアライズ
        let matrix_entries: Vec<MatrixEntry<N>> = self.matrix.iter()
            .map(|(vec, doc_len)| MatrixEntry::Full((vec.clone(), *doc_len)))
            .collect();
            
        let helper = IndexHelper {
            matrix: matrix_entries,
            doc_id: self.doc_id.clone(),
            corpus_token_freq: self.corpus_token_freq.clone(),
        };
        helper.serialize(serializer)
    }
}

impl<'de, N> Deserialize<'de> for Index<N>
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>
    {
        let helper = IndexHelper::<N>::deserialize(deserializer)?;
        // 各MatrixEntryが OnlyVec の場合は doc_len を1に
        let matrix = helper.matrix.into_iter()
            .map(|entry| entry.into())
            .collect();
        Ok(Index {
            matrix,
            doc_id: helper.doc_id,
            corpus_token_freq: helper.corpus_token_freq,
        })
    }
}

/// ヘルパー構造体を利用してIndex全体をシリアライズ／デシリアライズする
#[derive(Serialize, Deserialize)]
struct IndexHelper<N>
where
    N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded,
{
    matrix: Vec<MatrixEntry<N>>,
    doc_id: Vec<String>,
    corpus_token_freq: TokenFrequency,
}

// ... 既存のIndexの impl やその他のコード ...