use crate::{utils::math::vector::{ZeroSpVec, ZeroSpVecTrait}, vectorizer::{corpus::Corpus, token::TokenFrequency}};

pub trait TFIDFEngine<N>
where
    N: num::Num + Copy,
{
    /// IDFベクトルを生成するメソッド
    /// # Arguments
    /// * `corpus` - コーパス
    /// * `token_dim_sample` - トークンの次元サンプル
    /// # Returns
    /// * `Vec<N>` - IDFベクトル
    /// * `denormalize_num` - 正規化解除のための数値
    fn idf_vec(corpus: &Corpus, token_dim_sample: &[String]) -> (Vec<N>, f64);
    /// TFベクトルを生成するメソッド
    /// # Arguments
    /// * `freq` - トークン頻度
    /// * `token_dim_sample` - トークンの次元サンプル
    /// # Returns
    /// * `(ZeroSpVec<N>, f64)` - TFベクトルと正規化解除のための数値
    fn tf_vec(freq: &TokenFrequency, token_dim_sample: &[String]) -> (ZeroSpVec<N>, f64);
    /// TF-IDFを計算するイテレータ
    /// # Arguments
    /// * `tf` - TFベクトルのイテレータ
    /// * `tf_denorm` - TFの正規化解除のための数値
    /// * `idf` - IDFベクトルのイテレータ
    /// * `idf_denorm` - IDFの正規化解除のための数値
    /// # Returns
    /// * `(impl Iterator<Item = N>, f64)` - TF-IDFのイテレータと正規化解除のための数値
    /// 
    /// tfidfのdenormは tf idf ともにmaxが 1.0 のはずなので tf_denorm * idf_denorm で計算できる(intでの計算くそめんどいやつ)
    fn tfidf_iter_calc(tf: impl Iterator<Item = N>, tf_denorm: f64, idf: impl Iterator<Item = N>, idf_denorm: f64) -> (impl Iterator<Item = N>, f64);
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

    fn tfidf_iter_calc(tf: impl Iterator<Item = f32>, tf_denorm: f64, idf: impl Iterator<Item = f32>, idf_denorm: f64) -> (impl Iterator<Item = f32>, f64) {
        let tfidf = tf.zip(idf).map(move |(tf_val, idf_val)| {
            let tfidf = tf_val * idf_val;
            tfidf
        });
        (tfidf, tf_denorm * idf_denorm)
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

    fn tfidf_iter_calc(tf: impl Iterator<Item = f64>, tf_denorm: f64, idf: impl Iterator<Item = f64>, idf_denorm: f64) -> (impl Iterator<Item = f64>, f64) {
        let tfidf = tf.zip(idf).map(move |(tf_val, idf_val)| {
            let tfidf = tf_val * idf_val;
            tfidf
        });
        (tfidf, tf_denorm * idf_denorm)
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

    fn tfidf_iter_calc(tf: impl Iterator<Item = u32>, tf_denorm: f64, idf: impl Iterator<Item = u32>, idf_denorm: f64) -> (impl Iterator<Item = u32>, f64) {
        // denormのコストを考える
        // (tf_val / u32::MAX) * tf_denorm 
        // 除算遅いから
        // const base = 1 / u32::MAX as f64;
        // (tf_val * base) * tf_denorm
        // で計算する
        // 合計5回の乗算
        // を const val = base * tf_denorm * base * idf_denorm
        // で (tf * idf * val)
        // でf64 生の値が出る
        // を0-1に正規化 楽観的なmaxとしてtf_denorm * idf_denorm
        // tf * idf * (1 / u32::MAX) * (1 / u32::MAX) * tf_denorm * idf_denorm / (tf_denorm * idf_denorm) * u32::MAX 
        // tf * idf * (1 / u32::MAX)
        // done
        let tfidf = tf.zip(idf).map(move |(tf_val, idf_val)| {
            let tfidf = (tf_val as u64 * idf_val as u64) / u32::MAX as u64;
            tfidf as u32
        });
        (tfidf, tf_denorm * idf_denorm)
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

    fn tfidf_iter_calc(tf: impl Iterator<Item = u16>, tf_denorm: f64, idf: impl Iterator<Item = u16>, idf_denorm: f64) -> (impl Iterator<Item = u16>, f64) {
        let tfidf = tf.zip(idf).map(move |(tf_val, idf_val)| {
            let tfidf = (tf_val as u32 * idf_val as u32) / u16::MAX as u32;
            tfidf as u16
        });
        (tfidf, tf_denorm * idf_denorm)
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

    fn tfidf_iter_calc(tf: impl Iterator<Item = u8>, tf_denorm: f64, idf: impl Iterator<Item = u8>, idf_denorm: f64) -> (impl Iterator<Item = u8>, f64) {
        let tfidf = tf.zip(idf).map(move |(tf_val, idf_val)| {
            let tfidf = (tf_val as u32 * idf_val as u32) / u8::MAX as u32;
            tfidf as u8
        });
        (tfidf, tf_denorm * idf_denorm)
    }
}







