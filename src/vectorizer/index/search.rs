use std::ops::{AddAssign, MulAssign};

use num::Num;

use crate::{utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}}, TokenFrequency};
use rayon::prelude::*;

use super::Index;

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded, f64: IntoNormalizer<N> {
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
    #[inline]
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

        let idf_vec = 
        self.corpus_token_freq
            .idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();

        // TF-IDFの計算
        vec = vec.hadamard_normalized_vec(&idf_vec);
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
        for (i, (doc_vec, _)) in self.matrix.iter().enumerate() {
            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
            let similarity = tf_idf_doc_vec.cosine_similarity_normalized::<f64>(&idf_query);
            if similarity != 0.0 {
                result.push((i, similarity));
            }
        }

        // 類似度でソート
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut final_result: Vec<(String, f64)> = Vec::with_capacity(result.len());
        final_result.extend(result.into_iter().map(|(i, sim)| (self.doc_id[i].clone(), sim)));
        final_result
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
    #[inline]
    pub fn search_cosine_similarity_parallel(&self, tf_idf_query: &ZeroSpVec<N>, thread_count: usize) -> Vec<(String, f64)>
    where
        N: Send + Sync,
    {
        let idf_vec = 
            self.corpus_token_freq
                .idf_vector_ref_str::<N>(self.matrix.len() as u64)
                .into_iter()
                .map(|(_, idf)| idf)
                .collect::<Vec<N>>();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("Failed to build thread pool");

        let chunk_size = ((self.matrix.len() + thread_count - 1) / thread_count) / 4; // スレッド数で割り切れない場合は、余りを考慮して調整

        let mut result: Vec<(usize, f64)> = pool.install(|| {
            self.matrix
                .par_chunks(chunk_size)
                .enumerate()
                .flat_map(|(chunk_idx, chunk)| {
                    let base_idx = chunk_idx * chunk_size;
                    chunk
                        .iter()
                        .enumerate()
                        .filter_map(|(i, (doc_vec, _))| {
                            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
                            let similarity = tf_idf_doc_vec.cosine_similarity_normalized::<f64>(&tf_idf_query);
                            if similarity != 0.0 {
                                Some((base_idx + i, similarity))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });

        // 類似度でソート
        result.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut final_result: Vec<(String, f64)> = Vec::with_capacity(result.len());
        final_result.extend(result.into_iter().map(|(i, sim)| (self.doc_id[i].clone(), sim)));
        final_result
    }

    /// クエリを検索するメソッド
    /// dot積を計算し、類似度の高い順にソートして返します。
    #[inline]
    pub fn search_dot(&self, tf_idf_query: &ZeroSpVec<N>) -> Vec<(String, f64)> {
        let mut result: Vec<(usize, f64)> = Vec::new();

        let idf_vec = 
        self.corpus_token_freq
            .idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();

        // ドキュメントベクトルとqueryを掛け算してコサイン類似度を計算
        for (i, (doc_vec, _)) in self.matrix.iter().enumerate() {
            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
            let similarity = tf_idf_doc_vec.dot_normalized::<f64>(tf_idf_query);
            if similarity != 0.0 {
                result.push((i, similarity.into()));
            }
        }

        // 類似度でソート
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut final_result: Vec<(String, f64)> = Vec::with_capacity(result.len());
        final_result.extend(result.into_iter().map(|(i, sim)| (self.doc_id[i].clone(), sim)));
        final_result
    }

    pub fn search_dot_paeallel(&self, tf_idf_query: &ZeroSpVec<N>, thread_count: usize) -> Vec<(String, f64)>
    where
        N: Send + Sync,
    {
        let idf_vec = 
            self.corpus_token_freq
                .idf_vector_ref_str::<N>(self.matrix.len() as u64)
                .into_iter()
                .map(|(_, idf)| idf)
                .collect::<Vec<N>>();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("Failed to build thread pool");

        let chunk_size = ((self.matrix.len() + thread_count - 1) / thread_count) / 4; // スレッド数で割り切れない場合は、余りを考慮して調整

        let mut result: Vec<(usize, f64)> = pool.install(|| {
            self.matrix
                .par_chunks(chunk_size)
                .enumerate()
                .flat_map(|(chunk_idx, chunk)| {
                    let base_idx = chunk_idx * chunk_size;
                    chunk
                        .iter()
                        .enumerate()
                        .filter_map(|(i, (doc_vec, _))| {
                            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
                            let similarity = tf_idf_doc_vec.dot_normalized::<f64>(tf_idf_query);
                            if similarity != 0.0 {
                                Some((base_idx + i, similarity.into()))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });

        // 類似度でソート
        result.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut final_result: Vec<(String, f64)> = Vec::with_capacity(result.len());
        final_result.extend(result.into_iter().map(|(i, sim)| (self.doc_id[i].clone(), sim)));
        final_result
    }
}