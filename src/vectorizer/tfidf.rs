
use num_traits::Num;

use crate::{utils::datastruct::{map::IndexSet, vector::{ZeroSpVec, ZeroSpVecTrait}}, vectorizer::{corpus::Corpus, token::TokenFrequency}};

pub trait TFIDFEngine<N, K>: Send + Sync
where
    N: Num + Copy,
{
    /// Method to generate the IDF vector
    /// # Arguments
    /// * `corpus` - The corpus
    /// * `token_dim_sample` - Token dimension sample
    /// # Returns
    /// * `Vec<N>` - The IDF vector
    /// * `denormalize_num` - Value for denormalization
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<N>, f64);
    /// Method to generate the TF vector
    /// # Arguments
    /// * `freq` - Token frequency
    /// * `token_dim_sample` - Token dimension sample
    /// # Returns
    /// * `(ZeroSpVec<N>, f64)` - TF vector and value for denormalization
    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<N>, f64);
}

/// デフォルトのTF-IDFエンジン
/// `f32`、`f64`、`u32`、`u16`、`u8`の型に対応
#[derive(Debug)]
pub struct DefaultTFIDFEngine;
impl DefaultTFIDFEngine {
    pub fn new() -> Self {
        DefaultTFIDFEngine
    }
}

impl<K> TFIDFEngine<f32, K> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<f32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample.iter() {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as f32);
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<f32>, f64) {
        // Build sparse TF vector: only non-zero entries are stored
        let total_count = freq.token_sum() as f32;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count.into()); }
        let len = token_dim_sample.len();
        let inv_total = 1.0f32 / total_count;
        let mut raw = freq.iter().map(|(token, count)| {
            let idx = token_dim_sample.get_index(token).unwrap();
            (idx, (count as f32) * inv_total)
        }).collect::<Vec<_>>();
        raw.sort_unstable_by_key(|(idx, _)| *idx);
        (unsafe { ZeroSpVec::from_sparse_iter(raw.into_iter(), len) }, total_count.into())
    }
}

impl<K> TFIDFEngine<f64, K> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<f64>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample.iter() {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<f64>, f64) {
        // Build sparse TF vector: only non-zero entries are stored
        let total_count = freq.token_sum() as f64;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count.into()); }
        let len = token_dim_sample.len();
        let inv_total = 1.0f64 / total_count;
        let mut raw = freq.iter().map(|(token, count)| {
            let idx = token_dim_sample.get_index(token).unwrap();
            (idx, (count as f64) * inv_total)
        }).collect::<Vec<_>>();
        raw.sort_unstable_by_key(|(idx, _)| *idx);
        (unsafe { ZeroSpVec::from_sparse_iter(raw.into_iter(), len) }, total_count.into())
    }
}

impl<K> TFIDFEngine<u32, K> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<u32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample.iter() {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        let max = idf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        (
        idf_vec
            .into_iter()
            .map(|idf| (idf / max * u32::MAX as f64).ceil() as u32)
            .collect(),
        max
        )
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<u32>, f64) {
        // Build sparse TF vector without allocating dense Vec
        let total_count = freq.token_sum() as f64;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count); }
        let mut max_val = 0.0f64;
        let inv_total = 1.0f64 / total_count;
        let mut raw: Vec<(usize, f64)> = freq.iter().map(|(token, count)| {
            let idx = token_dim_sample.get_index(token).unwrap();
            let v = (count as f64) * inv_total;
            max_val = max_val.max(v);
            (idx, v)
        }).collect::<Vec<_>>();
        let len = token_dim_sample.len();
        let mul_norm = (u32::MAX as f64) / max_val; // == (1/max_val) * u32::MAX
        let vec_u32 = raw.drain(..)
            .map(|(idx, v)| {
                let q = (v * mul_norm).ceil() as u32;
                (idx, q)
            })
            .collect::<Vec<_>>();
        (unsafe { ZeroSpVec::from_sparse_iter(vec_u32.into_iter(), len) }, total_count)
    }
}

impl<K> TFIDFEngine<u16, K> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<u16>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample.iter() {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        let max = idf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        (
        idf_vec
            .into_iter()
            .map(|idf| (idf / max * u16::MAX as f64).ceil() as u16)
            .collect(),
        max
        )
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<u16>, f64) {
        // Build sparse TF vector without allocating a dense Vec<f64>
        let total_count = freq.token_sum() as f64;
        // First pass: compute raw tf values and track max
        let mut max_val = 0.0f32;
        let div_total = (1.0 / total_count) as f32;
        let raw = freq.iter().map(|(token, count)| {
            let idx = token_dim_sample.get_index(token).unwrap();
            let v = (count as f32) * div_total;
            max_val = max_val.max(v);
            (idx, v)
        }).collect::<Vec<_>>();
        let len = token_dim_sample.len();
        // Second pass: normalize into quantized u16 and build sparse vector
    let norm_div_max = (u16::MAX as f32) / max_val; // == (1/max_val) * u16::MAX
        let vec_u16 = raw.into_iter()
            .map(|(idx, v)| {
                let q = (v * norm_div_max).ceil() as u16;
                (idx, q)
            })
            .collect::<Vec<_>>();
        (unsafe { ZeroSpVec::from_sparse_iter(vec_u16.into_iter(), len) }, total_count)
    }
}

impl<K> TFIDFEngine<u8, K> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &Vec<Box<str>>) -> (Vec<u8>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample.iter() {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        let max = idf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        (
        idf_vec
            .into_iter()
            .map(|idf| (idf / max * u8::MAX as f64).ceil() as u8)
            .collect(),
        max
        )
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<u8>, f64) {
        // Build sparse TF vector without allocating dense Vec
        let total_count_f64 = freq.token_sum() as f64;
        if total_count_f64 == 0.0 { return (ZeroSpVec::new(), total_count_f64); }
        // Use f32 intermediates for u8 to reduce cost and memory
        let total_count = total_count_f64 as f32;
        let mut max_val = 0.0f32;
        let inv_total = 1.0f32 / total_count;
        let raw = freq.iter().map(|(token, count)| {
            let idx = token_dim_sample.get_index(token).unwrap();
            let v = (count as f32) * inv_total;
            max_val = max_val.max(v);
            (idx, v)
        }).collect::<Vec<_>>();
        let len = token_dim_sample.len();
        if max_val == 0.0 { return (ZeroSpVec::new(), total_count_f64); }
        let mul_norm = (u8::MAX as f32) / max_val; // == (1/max_val) * u8::MAX
        let vec_u8 = raw.into_iter()
            .map(|(idx, v)| {
                let q = (v * mul_norm).ceil() as u8;
                (idx, q)
            })
            .collect::<Vec<_>>();
        (unsafe { ZeroSpVec::from_sparse_iter(vec_u8.into_iter(), len) }, total_count_f64)
    }
}