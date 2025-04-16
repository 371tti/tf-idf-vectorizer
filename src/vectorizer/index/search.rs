use std::ops::{AddAssign, MulAssign};

use num::Num;

use crate::{utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}}, TokenFrequency};
use rayon::{prelude::*, result};

use super::Index;

pub enum SearchMethod {
    CosineSimilarity,
    CosineSimilarityParallel(usize),
    DotProduct,
    DotProductParallel(usize),
}

pub enum SearchBias {
    None,
    LenPenalty(f64),
}

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded + Send + Sync, f64: IntoNormalizer<N> {
    pub fn search(&self, method: SearchMethod, query: &ZeroSpVec<N>, top_n: usize, parameter: SearchBias) -> Vec<(String, f64)> {
        let idf_vec = 
            self.corpus_token_freq.
            idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();
        
        let mut score_vec = match method {
            SearchMethod::CosineSimilarity => self.search_cosine_similarity(query, &idf_vec),
            SearchMethod::CosineSimilarityParallel(thread_count) => self.search_cosine_similarity_parallel(query, &idf_vec, thread_count),
            SearchMethod::DotProduct => self.search_dot(query, &idf_vec),
            SearchMethod::DotProductParallel(thread_count) => self.search_dot_parallel(query, &idf_vec, thread_count),
        };

        // パラメータを反映
        match parameter {
            SearchBias::None => {},
            SearchBias::LenPenalty(penalty) => {
                let min_doc_len = self.doc_token_count.iter().min().unwrap_or(&1u64);
                let max_doc_len = self.doc_token_count.iter().max().unwrap_or(&1u64);
                let avg_doc_len = self.corpus_token_freq.token_total_count() as f64 / self.doc_num() as f64;

                // 各スコアに対して、平均文書長と各文書長の比率に基づくバイアスを適用する
                score_vec.iter_mut().for_each(|(idx, score)| {
                    let doc_len = self.doc_token_count[*idx] as f64;
                    // 文書が平均より長い場合はスコアを低下させ、短い場合は上昇させる
                    *score *= (avg_doc_len / doc_len).powf(penalty);
                });
            }
        }

        // スコアをソートして上位top_nを取得
        let mut result = Vec::with_capacity(top_n);
        score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.extend(score_vec.into_iter().take(top_n).map(|(i, score)| (self.doc_id[i].clone(), score)));

        result
    }
    
    /// クエリを検索するメソッド
    /// コサイン類似度
    /// # Arguments
    /// * `query` - クエリのベクトル
    /// 
    /// # Returns
    /// * `Vec<(String, f64)>` - 検索結果のベクトル
    pub fn search_cosine_similarity(&self, query: &ZeroSpVec<N>, idf_vec: &Vec<N>) -> Vec<(usize, f64)> {
        let mut result = Vec::new();

        // IDFとqueryを先に乗算
        let idf_query: ZeroSpVec<N> = query.hadamard_normalized_vec(&idf_vec);

        // ドキュメントベクトルとIDFを掛け算してコサイン類似度を計算
        for (i, doc_vec) in self.matrix.iter().enumerate() {
            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
            let similarity = tf_idf_doc_vec.cosine_similarity_normalized::<f64>(&idf_query);
            if similarity != 0.0 {
                result.push((i, similarity));
            }
        }

        result
    }

    /// クエリを検索するメソッド
    /// コサイン類似度
    /// 並列処理を使用して、検索を高速化します。
    /// 
    /// # Arguments
    /// * `query` - クエリのベクトル
    /// * `thread_count` - スレッド数
    /// 
    /// # Returns
    /// * `Vec<(String, f64)>` - 検索結果のベクトル
    #[inline]
    pub fn search_cosine_similarity_parallel(&self, tf_idf_query: &ZeroSpVec<N>, idf_vec: &Vec<N>, thread_count: usize) -> Vec<(usize, f64)>
    where
        N: Send + Sync,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("Failed to build thread pool");

        let chunk_size = ((self.matrix.len() + thread_count - 1) / thread_count) / 4; // スレッド数で割り切れない場合は、余りを考慮して調整

        let result: Vec<(usize, f64)> = pool.install(|| {
            self.matrix
                .par_chunks(chunk_size)
                .enumerate()
                .flat_map(|(chunk_idx, chunk)| {
                    let base_idx = chunk_idx * chunk_size;
                    chunk
                        .iter()
                        .enumerate()
                        .filter_map(|(i, doc_vec)| {
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

        result
    }

    /// クエリを検索するメソッド
    /// dot積を計算
    #[inline]
    pub fn search_dot(&self, tf_idf_query: &ZeroSpVec<N>, idf_vec: &Vec<N>) -> Vec<(usize, f64)> {
        let mut result: Vec<(usize, f64)> = Vec::new();

        // ドキュメントベクトルとqueryを掛け算してドット積を計算
        for (i, doc_vec) in self.matrix.iter().enumerate() {
            let tf_idf_doc_vec = doc_vec.hadamard_normalized_vec(&idf_vec);
            let similarity = tf_idf_doc_vec.dot_normalized::<f64>(tf_idf_query);
            if similarity != 0.0 {
                result.push((i, similarity.into()));
            }
        }

        result
    }

    pub fn search_dot_parallel(&self, tf_idf_query: &ZeroSpVec<N>, idf_vec: &Vec<N>, thread_count: usize) -> Vec<(usize, f64)>
    where
        N: Send + Sync,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("Failed to build thread pool");

        let chunk_size = ((self.matrix.len() + thread_count - 1) / thread_count) / 4; // スレッド数で割り切れない場合は、余りを考慮して調整

        let result: Vec<(usize, f64)> = pool.install(|| {
            self.matrix
                .par_chunks(chunk_size)
                .enumerate()
                .flat_map(|(chunk_idx, chunk)| {
                    let base_idx = chunk_idx * chunk_size;
                    chunk
                        .iter()
                        .enumerate()
                        .filter_map(|(i, doc_vec)| {
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

        result
    }
}