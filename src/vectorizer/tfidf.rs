use ahash::RandomState;
use indexmap::IndexSet;

use crate::{utils::math::vector::{ZeroSpVec, ZeroSpVecTrait}, vectorizer::{corpus::Corpus, token::TokenFrequency}};

pub trait TFIDFEngine<N>: Send + Sync
where
    N: num::Num + Copy,
{
    /// Method to generate the IDF vector
    /// # Arguments
    /// * `corpus` - The corpus
    /// * `token_dim_sample` - Token dimension sample
    /// # Returns
    /// * `Vec<N>` - The IDF vector
    /// * `denormalize_num` - Value for denormalization
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<N>, f64);
    /// Method to generate the TF vector
    /// # Arguments
    /// * `freq` - Token frequency
    /// * `token_dim_sample` - Token dimension sample
    /// # Returns
    /// * `(ZeroSpVec<N>, f64)` - TF vector and value for denormalization
    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<N>, f64);
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

impl TFIDFEngine<f32> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<f32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as f32);
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<f32>, f64) {
        // Build sparse TF vector: only non-zero entries are stored
        let total_count = freq.token_sum() as f32;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count.into()); }
        let mut raw: Vec<(usize, f32)> = Vec::with_capacity(freq.token_num());
        let len = token_dim_sample.len();
        let inv_total = 1.0f32 / total_count;
        for (idx, token) in token_dim_sample.iter().enumerate() {
            let count = freq.token_count(token) as f32;
            if count == 0.0 { continue; }
            raw.push((idx, count * inv_total));
        }
        (unsafe { ZeroSpVec::from_raw_iter(raw.into_iter(), len) }, total_count.into())
    }
}

impl TFIDFEngine<f64> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<f64>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<f64>, f64) {
        // Build sparse TF vector: only non-zero entries are stored
        let total_count = freq.token_sum() as f64;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count.into()); }
        let mut raw: Vec<(usize, f64)> = Vec::with_capacity(freq.token_num());
        let len = token_dim_sample.len();
        let inv_total = 1.0f64 / total_count;
        for (idx, token) in token_dim_sample.iter().enumerate() {
            let count = freq.token_count(token) as f64;
            if count == 0.0 { continue; }
            raw.push((idx, count * inv_total));
        }
        (unsafe { ZeroSpVec::from_raw_iter(raw.into_iter(), len) }, total_count.into())
    }
}

impl TFIDFEngine<u32> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<u32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as u32);
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<u32>, f64) {
        // Build sparse TF vector without allocating dense Vec
        let total_count = freq.token_sum() as f64;
        if total_count == 0.0 { return (ZeroSpVec::new(), total_count); }
        let mut raw: Vec<(usize, f64)> = Vec::with_capacity(freq.token_num());
        let mut max_val = 0.0f64;
        let inv_total = 1.0f64 / total_count;
        for (idx, token) in token_dim_sample.iter().enumerate() {
            let count = freq.token_count(token) as f64;
            if count == 0.0 { continue; }
            let v = count * inv_total;
            if v > max_val { max_val = v; }
            raw.push((idx, v));
        }
        let len = token_dim_sample.len();
        if max_val == 0.0 { return (ZeroSpVec::new(), total_count); }
        let mut vec_u32: Vec<(usize, u32)> = Vec::with_capacity(raw.len());
        let mul_norm = (u32::MAX as f64) / max_val; // == (1/max_val) * u32::MAX
        for (idx, v) in raw.into_iter() {
            let q = (v * mul_norm).ceil() as u32;
            vec_u32.push((idx, q));
        }
        (unsafe { ZeroSpVec::from_raw_iter(vec_u32.into_iter(), len) }, total_count)
    }
}

impl TFIDFEngine<u16> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<u16>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
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

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<u16>, f64) {
        // Build sparse TF vector without allocating a dense Vec<f64>
        let total_count = freq.token_sum() as f64;
        // First pass: compute raw tf values and track max
        let mut raw: Vec<(usize, f32)> = Vec::with_capacity(freq.token_num());
        let mut max_val = 0.0f32;
    let div_total = (1.0 / total_count) as f32;
        for (idx, token) in token_dim_sample.iter().enumerate() {
            let count = freq.token_count(token);
            if count == 0 { continue; }
            let v = count as f32 * div_total;
            if v > max_val { max_val = v; }
            raw.push((idx, v));
        }
        let len = token_dim_sample.len();
        if max_val == 0.0 { return (ZeroSpVec::new(), total_count); }
        // Second pass: normalize into quantized u16 and build sparse vector
        let mut vec_u16: Vec<(usize, u16)> = Vec::with_capacity(raw.len());
    let norm_div_max = (u16::MAX as f32) / max_val; // == (1/max_val) * u16::MAX
        for (idx, v) in raw.into_iter() {
            let q = (v * norm_div_max).ceil() as u16;
            vec_u16.push((idx, q));
        }
        (unsafe { ZeroSpVec::from_raw_iter(vec_u16.into_iter(), len) }, total_count)
    }
}

impl TFIDFEngine<u8> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &IndexSet<String, RandomState>) -> (Vec<u8>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
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

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &IndexSet<String, RandomState>) -> (ZeroSpVec<u8>, f64) {
        // Build sparse TF vector without allocating dense Vec
        let total_count_f64 = freq.token_sum() as f64;
        if total_count_f64 == 0.0 { return (ZeroSpVec::new(), total_count_f64); }
        // Use f32 intermediates for u8 to reduce cost and memory
        let total_count = total_count_f64 as f32;
        let mut raw: Vec<(usize, f32)> = Vec::with_capacity(freq.token_num());
        let mut max_val = 0.0f32;
        let inv_total = 1.0f32 / total_count;
        for (idx, token) in token_dim_sample.iter().enumerate() {
            let count = freq.token_count(token) as f32;
            if count == 0.0 { continue; }
            let v = count * inv_total;
            if v > max_val { max_val = v; }
            raw.push((idx, v));
        }
        let len = token_dim_sample.len();
        if max_val == 0.0 { return (ZeroSpVec::new(), total_count_f64); }
        let mut vec_u8: Vec<(usize, u8)> = Vec::with_capacity(raw.len());
        let mul_norm = (u8::MAX as f32) / max_val; // == (1/max_val) * u8::MAX
        for (idx, v) in raw.into_iter() {
            let q = (v * mul_norm).ceil() as u8;
            vec_u8.push((idx, q));
        }
        (unsafe { ZeroSpVec::from_raw_iter(vec_u8.into_iter(), len) }, total_count_f64)
    }
}