use half::f16;
use num_traits::{Num, Pow};

use crate::{Corpus, TermFrequency, utils::datastruct::{map::IndexSet, vector::{TFVector, TFVectorTrait}}};


/// TF-IDF Calculation Engine Trait
///
/// Defines the behavior of a TF-IDF calculation engine.
///
/// Custom engines can be implemented and plugged into
/// [`TFIDFVectorizer`].
///
/// A default implementation, [`DefaultTFIDFEngine`], is provided.
///
/// ### Supported Numeric Types
/// - `f16`
/// - `f32`
/// - `u16`
/// - `u32`
pub trait TFIDFEngine<N>: Send + Sync
where
    N: Num + Copy
{
    /// Method to generate the IDF vector
    /// # Arguments
    /// * `corpus` - The corpus
    /// * `term_dim_sample` - term dimension sample
    /// # Returns
    /// * `Vec<N>` - The IDF vector
    /// * `denormalize_num` - Value for denormalization
    fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> Vec<f32> {
        let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for term in term_dim_sample.iter() {
            let doc_freq = corpus.get_term_count(term);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as f32);
        }
        idf_vec
    }
    /// Method to generate the TF vector
    /// # Arguments
    /// * `freq` - term frequency
    /// * `term_dim_sample` - term dimension sample
    /// # Returns
    /// * `(ZeroSpVec<N>, f64)` - TF vector and value for denormalization
    fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> TFVector<N>;

    fn tf_denorm(val: N) -> u32;
}

/// デフォルトのTF-IDFエンジン
#[derive(Debug)]
pub struct DefaultTFIDFEngine;
impl DefaultTFIDFEngine {
    pub fn new() -> Self {
        DefaultTFIDFEngine
    }
}

impl TFIDFEngine<f16> for DefaultTFIDFEngine {
    // fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> (Vec<f16>, f64) {
    //     let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
    //     let doc_num = corpus.get_doc_num() as f64;
    //     for term in term_dim_sample.iter() {
    //         let doc_freq = corpus.get_term_count(term);
    //         idf_vec.push(f16::from_f64(doc_num / (doc_freq as f64 + 1.0)));
    //     }
    //     (idf_vec, 1.0)
    // }
    #[inline]
    fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> TFVector<f16> {
        // Build sparse TF vector: only non-zero entries are stored
        let term_sum = freq.term_sum() as u32;
        let len = freq.term_num();
        let mut ind_vec: Vec<u32> = Vec::with_capacity(len);
        let mut val_vec: Vec<f16> = Vec::with_capacity(len);
        for (term, count) in freq.iter() {
            let count = (count as f32).sqrt();
            if let Some(idx) = term_dim_sample.get_index(term) {
                ind_vec.push(idx as u32);
                val_vec.push(f16::from_f32(count));
            }
        }
        unsafe { TFVector::from_vec(ind_vec, val_vec, len as u32, term_sum) }
    }

    #[inline(always)]
    fn tf_denorm(val: f16) -> u32 {
        val.to_f32().pow(2) as u32
    }
}

impl TFIDFEngine<f32> for DefaultTFIDFEngine
{
    // fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> (Vec<f32>, f64) {
    //     let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
    //     let doc_num = corpus.get_doc_num() as f64;
    //     for term in term_dim_sample.iter() {
    //         let doc_freq = corpus.get_term_count(term);
    //         idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as f32);
    //     }
    //     (idf_vec, 1.0)
    // }
    #[inline]
    fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> TFVector<f32> {
        // Build sparse TF vector: only non-zero entries are stored
        let term_sum = freq.term_sum() as u32;
        let len = freq.term_num();
        let mut ind_vec: Vec<u32> = Vec::with_capacity(len);
        let mut val_vec: Vec<f32> = Vec::with_capacity(len);
        for (term, count) in freq.iter() {
            if let Some(idx) = term_dim_sample.get_index(term) {
                ind_vec.push(idx as u32);
                val_vec.push(count as f32);
            }
        }
        unsafe { TFVector::from_vec(ind_vec, val_vec, len as u32, term_sum) }
    }

    #[inline(always)]
    fn tf_denorm(val: f32) -> u32 {
        val as u32
    }
}

impl TFIDFEngine<u32> for DefaultTFIDFEngine
{
    // fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> (Vec<u32>, f64) {
    //     let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
    //     let doc_num = corpus.get_doc_num() as f64;
    //     for term in term_dim_sample.iter() {
    //         let doc_freq = corpus.get_term_count(term);
    //         idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
    //     }
    //     let max = idf_vec
    //         .iter()
    //         .max_by(|a, b| a.total_cmp(b))
    //         .copied()
    //         .unwrap_or(1.0);
    //     (
    //     idf_vec
    //         .into_iter()
    //         .map(|idf| (idf / max * u32::MAX as f64).ceil() as u32)
    //         .collect(),
    //     max
    //     )
    // }
    #[inline]
    fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> TFVector<u32> {
        // Build sparse TF vector: only non-zero entries are stored
        let term_sum = freq.term_sum() as u32;
        let len = freq.term_num();
        let mut ind_vec: Vec<u32> = Vec::with_capacity(len);
        let mut val_vec: Vec<u32> = Vec::with_capacity(len);
        for (term, count) in freq.iter() {
            if let Some(idx) = term_dim_sample.get_index(term) {
                ind_vec.push(idx as u32);
                val_vec.push(count as u32);
            }
        }
        unsafe { TFVector::from_vec(ind_vec, val_vec, len as u32, term_sum) }
    }

    #[inline(always)]
    fn tf_denorm(val: u32) -> u32 {
        val
    }
}

impl TFIDFEngine<u16> for DefaultTFIDFEngine
{
    // fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> (Vec<u16>, f64) {
    //     let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
    //     let doc_num = corpus.get_doc_num() as f64;
    //     for term in term_dim_sample.iter() {
    //         let doc_freq = corpus.get_term_count(term);
    //         idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
    //     }
    //     let max = idf_vec
    //         .iter()
    //         .max_by(|a, b| a.total_cmp(b))
    //         .copied()
    //         .unwrap_or(1.0);
    //     (
    //     idf_vec
    //         .into_iter()
    //         .map(|idf| (idf / max * u16::MAX as f64).ceil() as u16)
    //         .collect(),
    //     max
    //     )
    // }
    #[inline]
    fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> TFVector<u16> {
        // Build sparse TF vector: only non-zero entries are stored
        let term_sum = freq.term_sum() as u32;
        let len = freq.term_num();
        let mut ind_vec: Vec<u32> = Vec::with_capacity(len);
        let mut val_vec: Vec<u16> = Vec::with_capacity(len);
        for (term, count) in freq.iter() {
            if let Some(idx) = term_dim_sample.get_index(term) {
                ind_vec.push(idx as u32);
                val_vec.push(count as u16);
            }
        }
        unsafe { TFVector::from_vec(ind_vec, val_vec, len as u32, term_sum) }
    }

    #[inline(always)]
    fn tf_denorm(val: u16) -> u32 {
        val as u32
    }
}

// impl<K> TFIDFEngine<u8, K> for DefaultTFIDFEngine
// {
//     // fn idf_vec(corpus: &Corpus, term_dim_sample: &Vec<Box<str>>) -> (Vec<u8>, f64) {
//     //     let mut idf_vec = Vec::with_capacity(term_dim_sample.len());
//     //     let doc_num = corpus.get_doc_num() as f64;
//     //     for term in term_dim_sample.iter() {
//     //         let doc_freq = corpus.get_term_count(term);
//     //         idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
//     //     }
//     //     let max = idf_vec
//     //         .iter()
//     //         .max_by(|a, b| a.total_cmp(b))
//     //         .copied()
//     //         .unwrap_or(1.0);
//     //     (
//     //     idf_vec
//     //         .into_iter()
//     //         .map(|idf| (idf / max * u8::MAX as f64).ceil() as u8)
//     //         .collect(),
//     //     max
//     //     )
//     // }

//     fn tf_vec(freq: &TermFrequency, term_dim_sample: &IndexSet<Box<str>>) -> (ZeroSpVec<u8>, f32) {
//         // Build sparse TF vector without allocating dense Vec
//         let total_count_f64 = freq.term_sum() as f64;
//         if total_count_f64 == 0.0 { return (ZeroSpVec::new(), total_count_f64 as f32); }
//         // Use f32 intermediates for u8 to reduce cost and memory
//         let total_count = total_count_f64 as f32;
//         let mut max_val = 0.0f32;
//         let inv_total = 1.0f32 / total_count;
//         let raw = freq.iter().filter_map(|(term, count)| {
//             let idx = term_dim_sample.get_index(term)?;
//             let v = (count as f32) * inv_total;
//             max_val = max_val.max(v);
//             Some((idx, v))
//         }).collect::<Vec<_>>();
//         let len = term_dim_sample.len();
//         if max_val == 0.0 { return (ZeroSpVec::new(), total_count_f64 as f32); }
//         let mul_norm = (u8::MAX as f32) / max_val; // == (1/max_val) * u8::MAX
//         let vec_u8 = raw.into_iter()
//             .map(|(idx, v)| {
//                 let q = (v * mul_norm).ceil() as u8;
//                 (idx, q)
//             })
//             .collect::<Vec<_>>();
//         (unsafe { ZeroSpVec::from_sparse_iter(vec_u8.into_iter(), len) }, total_count_f64 as f32)
//     }
// }