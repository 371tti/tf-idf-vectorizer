
use crate::{utils::math::vector::{ZeroSpVec, ZeroSpVecTrait}, vectorizer::{corpus::Corpus, token::TokenFrequency}};

pub trait TFIDFEngine<N>
where
    N: num::Num,
{
    /// IDFベクトルを生成するメソッド
    /// # Arguments
    /// * `corpus` - コーパス
    /// * `token_dim_sample` - トークンの次元サンプル
    /// # Returns
    /// * `Vec<N>` - IDFベクトル
    /// * `denormalize_num` - 正規化解除のための数値
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<N>, f64);
    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<N>, f64);
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
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<f32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as f32);
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<f32>, f64) {
        let mut tf_vec = ZeroSpVec::with_capacity(token_dim_sample.len());
        let total_count = freq.token_sum() as f32;
        for token in token_dim_sample {
            let count = freq.token_count(token) as f32;
            tf_vec.push(count / total_count);
        }
        (tf_vec, total_count.into())
    }
}

impl TFIDFEngine<f64> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<f64>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push(doc_num / (doc_freq as f64 + 1.0));
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<f64>, f64) {
        let mut tf_vec = ZeroSpVec::with_capacity(token_dim_sample.len());
        let total_count = freq.token_sum() as f64;
        for token in token_dim_sample {
            let count = freq.token_count(token) as f64;
            tf_vec.push(count / total_count);
        }
        (tf_vec, total_count.into())
    }
}

impl TFIDFEngine<u32> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<u32>, f64) {
        let mut idf_vec = Vec::with_capacity(token_dim_sample.len());
        let doc_num = corpus.get_doc_num() as f64;
        for token in token_dim_sample {
            let doc_freq = corpus.get_token_count(token);
            idf_vec.push((doc_num / (doc_freq as f64 + 1.0)) as u32);
        }
        (idf_vec, 1.0)
    }

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<u32>, f64) {
        let mut tf_vec: Vec<f64> = Vec::with_capacity(token_dim_sample.len());
        let total_count = freq.token_sum() as f64;
        for token in token_dim_sample {
            let count = freq.token_count(token) as f64;
            tf_vec.push(count / total_count);
        }
        let max = tf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        let normalized_vec: Vec<u32> = tf_vec
            .into_iter()
            .map(|tf| (tf / max * u32::MAX as f64).ceil() as u32)
            .collect();
        (
            ZeroSpVec::from(normalized_vec),
            total_count
        )
    }
}

impl TFIDFEngine<u16> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<u16>, f64) {
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

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<u16>, f64) {
        let mut tf_vec: Vec<f64> = Vec::with_capacity(token_dim_sample.len());
        let total_count = freq.token_sum() as f64;
        for token in token_dim_sample {
            let count = freq.token_count(token) as f64;
            tf_vec.push(count / total_count);
        }
        let max = tf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        let normalized_vec: Vec<u16> = tf_vec
            .into_iter()
            .map(|tf| (tf / max * u16::MAX as f64).ceil() as u16)
            .collect();
        (
            ZeroSpVec::from(normalized_vec),
            total_count
        )
    }
}

impl TFIDFEngine<u8> for DefaultTFIDFEngine
{
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<u8>, f64) {
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

    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<u8>, f64) {
        let mut tf_vec: Vec<f64> = Vec::with_capacity(token_dim_sample.len());
        let total_count = freq.token_sum() as f64;
        for token in token_dim_sample {
            let count = freq.token_count(token) as f64;
            tf_vec.push(count / total_count);
        }
        let max = tf_vec
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .copied()
            .unwrap_or(1.0);
        let normalized_vec: Vec<u8> = tf_vec
            .into_iter()
            .map(|tf| (tf / max * u8::MAX as f64).ceil() as u8)
            .collect();
        (
            ZeroSpVec::from(normalized_vec),
            total_count
        )
    }
}







