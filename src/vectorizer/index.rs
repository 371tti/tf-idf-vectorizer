use std::ops::{AddAssign, MulAssign};

use num::Num;
use serde::{Deserialize, Serialize};

use crate::utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}};

use super::token::TokenFrequency;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded {
    pub matrix: Vec<ZeroSpVec<N>>,
    pub doc_id: Vec<String>,
    pub corpus_token_freq: TokenFrequency,
}

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded, f64: IntoNormalizer<N> {
    pub fn new() -> Self {
        Self {
            matrix: Vec::new(),
            doc_id: Vec::new(),
            corpus_token_freq: TokenFrequency::new(),
        }
    }

    /// インデックスにドキュメントを追加するメソッド
    pub fn add_doc(&mut self, doc_id: String, tokens: &[&str]) {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        // corpus_token_freqに追加
        let old_corpus_token_num = self.corpus_token_freq.token_num();
        self.corpus_token_freq.add_tokens(tokens);
        let added_corpus_token_num = self.corpus_token_freq.token_num() - old_corpus_token_num;
        self.doc_id.push(doc_id);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(doc_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = doc_tf.tf_token(token); // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }

        if added_corpus_token_num > 0 {
            // 新しいトークンが追加された場合、matrixを拡張する
            for other_tf in self.matrix.iter_mut() {
                other_tf.add_dim(added_corpus_token_num);
            }
        }

        // matrixに追加
        self.matrix.push(vec);
    }

    pub fn generate_query_mask(&self, tokens: &[&str]) -> ZeroSpVec<N> {
        // TFの計算
        let mut query_tf = TokenFrequency::new();
        query_tf.add_tokens(tokens);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(query_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = if query_tf.contains_token(token) { N::max_normalized() } else { N::zero() }; // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }
        vec
    }

    pub fn generate_query(&self, tokens: &[&str]) -> ZeroSpVec<N> {
        // TFの計算
        let mut query_tf = TokenFrequency::new();
        query_tf.add_tokens(tokens);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(query_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = query_tf.tf_token(token); // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }
        vec
    }

    pub fn search_cosine_similarity(&self, query: &ZeroSpVec<N>) -> Vec<(String, f64)> {
        let mut result = Vec::new();

        // IDFとqueryを先に乗算
        let idf_query: ZeroSpVec<N> = query.hadamard_normalized_vec(
            &self.corpus_token_freq.
            idf_vector_ref_str::<N>(self.matrix.len() as u64).into_iter().map(|(_, idf)| idf).collect::<Vec<N>>()
        );

        // ドキュメントベクトルとIDFを掛け算してコサイン類似度を計算
        for (i, doc_vec) in self.matrix.iter().enumerate() {
            let similarity = doc_vec.cosine_similarity_normalized(&idf_query);
            result.push((self.doc_id[i].clone(), similarity.into()));
        }

        // 類似度でソート
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        result
    }
}