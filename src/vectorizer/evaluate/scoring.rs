use std::fmt::Debug;

use num::Num;

use crate::{utils::{math::vector::{ZeroSpVec, ZeroSpVecTrait}, normalizer::DeNormalizer}, vectorizer::{compute::compare::{Compare, DefaultCompare}, tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer}};

/// 検索クエリの種類を定義する列挙型
pub enum SimilarityQuery {
    /// 内積
    /// 向きと大きさを考慮した類似度
    Dot(TokenFrequency),
    /// コサイン類似度
    /// 向きのみを考慮した類似度
    CosineSimilarity(TokenFrequency),
    /// BM25
    /// ドキュメントの長さを考慮した類似度
    BM25(TokenFrequency, f64, f64), // (k1, b)
}

/// 検索結果を格納する構造体
pub struct Hits<K> 
{
    pub list: Vec<(K, f64)>,
}

impl<K> Hits<K> {
    pub fn new(vec: Vec<(K, f64)>) -> Self {
        Hits { list: vec }
    }

    pub fn sort_by_score(&mut self) -> &mut Self {
    // NaN を除外 (必要なら末尾へ送る運用も可)
    self.list.retain(|(_, s)| !s.is_nan());
    // total_cmp で反射律/推移律を満たす全順序 (NaN 排除済みなので安全)
    self.list.sort_by(|a, b| b.1.total_cmp(&a.1));
    self
    }

    pub fn sort_by_score_rev(&mut self) -> &mut Self{
        // NaN を除外 (必要なら末尾へ送る運用も可)
        self.list.retain(|(_, s)| !s.is_nan());
        // total_cmp で反射律/推移律を満たす全順序 (NaN 排除済みなので安全)
        self.list.sort_by(|a, b| a.1.total_cmp(&b.1));
        self
    }
}

impl<K> Debug for Hits<K>
where
    K: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // Pretty print with alternate formatting: each hit on a new line
            writeln!(f, "Hits [")?;
            for (key, score) in &self.list {
                writeln!(f, "    {:?}: {:.6}", key, score)?;
            }
            write!(f, "]")
        } else {
            // Default debug output
            f.debug_list().entries(&self.list).finish()
        }
    }
}

impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    K: Clone,
    N: Num + Copy + Into<f64> + DeNormalizer,
    E: TFIDFEngine<N>,
    DefaultCompare: Compare<N>,
{
    pub fn similarity(&self, query: SimilarityQuery) -> Hits<K> {
        let result = match query {
            SimilarityQuery::Dot(freq) => self.scoring_dot(freq),
            SimilarityQuery::CosineSimilarity(freq) => self.scoring_cosine(freq),
            SimilarityQuery::BM25(freq, k1, b) => self.scoring_bm25(freq, k1, b),
        };

        Hits { list: result }
    }
}

/// 検索のHL実装
impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    K: Clone,
    N: Num + Copy + Into<f64> + DeNormalizer,
    E: TFIDFEngine<N>,
    DefaultCompare: Compare<N>,
{
    fn scoring_dot(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        // まずtfをTF vector に transform して
        // それから検索域の次元にあわせてIDFを作成
        // クエリのtfidfを計算
        // それを使ってドキュメントのTFIDFとドット積を計算
        // tf1 * tf2 * idf^2 
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, query_denorm) =
            E::tfidf_iter_calc(tf.iter().copied(), tf_denormalize_num, idf.iter().copied(), idf_denormalize_num);
        let query_vec: Vec<N> = query_iter.collect();

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, doc_denorm) =
                E::tfidf_iter_calc(doc.tf_vec.iter().copied(), doc.denormalize_num, idf.iter().copied(), idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::dot(query_vec.iter().copied(), doc_iter);
            let score = dot * query_denorm * doc_denorm;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_cosine(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, _query_denorm) =
            E::tfidf_iter_calc_sparse(tf.raw_iter().map(|(idx, val)| (idx, *val)), tf_denormalize_num, &idf, idf_denormalize_num);
        let query_vec: ZeroSpVec<N> = ZeroSpVec::from_raw_iter(query_iter);

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, _doc_denorm) =
                E::tfidf_iter_calc_sparse(doc.tf_vec.raw_iter().map(|(idx, val)| (idx, *val)), doc.denormalize_num, &idf, idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::cosine_similarity(query_vec.raw_iter().map(|(idx, val)| (idx, *val)), doc_iter);
            let score = dot;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_bm25(&self, freq: TokenFrequency, k1: f64, b: f64) -> Vec<(K, f64)> {
        let (tf, _tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let k1_p = k1 + 1.0;
        let rev_avg_p = self.documents.iter().map(|doc| doc.token_sum as f64).sum::<f64>() / self.documents.len() as f64;

        let doc_scores = self.documents.iter().map(|doc| {(
            doc.key.clone(),
            tf.raw_iter().map(|(idx, _val)| {
                let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).into();
                let tf: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                idf * ((tf * k1_p) / (tf + k1 * (1.0 - b + (b * rev_avg_p))))
            }).sum::<f64>()
        )}).collect();

        doc_scores
    }
}