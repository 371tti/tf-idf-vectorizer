use std::ops::{AddAssign, MulAssign};

use num::Num;
use serde::{Deserialize, Serialize};

use crate::utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}};
use rayon::prelude::*;

use super::token::TokenFrequency;

/// インデックス
/// ドキュメント単位でインデックスを作成、検索するための構造体です
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded {
    matrix: Vec<ZeroSpVec<N>>,
    doc_id: Vec<String>,
    corpus_token_freq: TokenFrequency,
}

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded, f64: IntoNormalizer<N> {
    /// 新しいインデックスを作成するメソッド
    pub fn new() -> Self {
        Self {
            matrix: Vec::new(),
            doc_id: Vec::new(),
            corpus_token_freq: TokenFrequency::new(),
        }
    }

    /// インデックスのドキュメント数を取得するメソッド
    pub fn doc_num(&self) -> usize {
        self.matrix.len()
    }

    /// インデックスのトークン数を取得するメソッド
    /// トークン数はユニークなトークンの数を返す
    pub fn token_num(&self) -> usize {
        self.corpus_token_freq.token_num()
    }

    /// インデックスにドキュメントを追加するメソッド
    /// 
    /// # Arguments
    /// * `doc_id` - ドキュメントのID
    /// * `tokens` - ドキュメントのトークン
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
        vec.shrink_to_fit();
        // matrixに追加
        self.matrix.push(vec);
    }

    /// query vectorを生成するメソッド
    /// 重要度を考慮せず、トークンの有無だけでベクトルを生成します。
    /// 
    /// # Arguments
    /// * `tokens` - クエリのトークン
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - クエリのベクトル
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

    /// query vectorを生成するメソッド
    /// 
    /// # Arguments
    /// * `tokens` - クエリのトークン
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - クエリのベクトル
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

    /// クエリを検索するメソッド
    /// コサイン類似度を計算し、類似度の高い順にソートして返します。
    /// # Arguments
    /// * `query` - クエリのベクトル
    /// 
    /// # Returns
    /// * `Vec<(String, f64)>` - 検索結果のベクトル
    pub fn search_cosine_similarity(&self, query: &ZeroSpVec<N>) -> Vec<(String, f64)> {
        let mut result = Vec::new();

        let idf_vec = 
            self.corpus_token_freq.
            idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();

        // IDFとqueryを先に乗算
        let idf_query: ZeroSpVec<N> = query.hadamard_normalized_vec(&idf_vec);

        // ドキュメントベクトルとIDFを掛け算してコサイン類似度を計算
        for (i, doc_vec) in self.matrix.iter().enumerate() {
            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
            let similarity = tf_idf_doc_vec.cosine_similarity_normalized::<f64>(&idf_query);
            if similarity != 0.0 {
                result.push((self.doc_id[i].clone(), similarity));
            }
        }

        // 類似度でソート
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        result
    }

    /// クエリを検索するメソッド
    /// コサイン類似度を計算し、類似度の高い順にソートして返します。
    /// 並列処理を使用して、検索を高速化します。
    /// 
    /// # Arguments
    /// * `query` - クエリのベクトル
    /// * `thread_count` - スレッド数
    /// 
    /// # Returns
    /// * `Vec<(String, f64)>` - 検索結果のベクトル
    pub fn search_cosine_similarity_parallel(&self, query: &ZeroSpVec<N>, thread_count: usize) -> Vec<(String, f64)>
    where
        N: Send + Sync,
    {
        let idf_vec = 
            self.corpus_token_freq.
            idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();

        // IDFとqueryを先に乗算
        let idf_query: ZeroSpVec<N> = query.hadamard_normalized_vec(&idf_vec);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("Failed to build thread pool");

        let mut result: Vec<(String, f64)> = pool.install(|| {
            self.matrix
                .par_iter()
                .enumerate()
                .filter_map(|(i, doc_vec)| {
                    let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
                    let similarity = tf_idf_doc_vec.cosine_similarity_normalized::<f64>(&idf_query);
                    if similarity != 0.0 {
                        Some((self.doc_id[i].clone(), similarity))
                    } else {
                        None
                    }
                })
                .collect()
        });

        // 類似度でソート
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        result
    }
}