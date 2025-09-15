use std::{cmp::Ordering, fmt::Debug};

use num::{pow::Pow, Num};

use crate::{utils::{math::vector::ZeroSpVecTrait, normalizer::DeNormalizer}, vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer}};

/// 検索クエリの種類を定義する列挙型
pub enum SimilarityQuery {
    /// dot product
    /// 向きと大きさを考慮した類似度
    Dot(TokenFrequency),
    /// cosine similarity
    /// 向きのみを考慮した類似度
    CosineSimilarity(TokenFrequency),
    /// BM25
    /// ドキュメントの長さを考慮した類似度
    /// param k1: term frequencyの飽和を制御するパラメータ
    /// param b: ドキュメント長の正規化を制御するパラメータ
    BM25(TokenFrequency, f64, f64), // (k1, b)
}

/// 検索結果を格納する構造体
pub struct Hits<K> 
{
    /// (ドキュメントID, スコア, ドキュメント長)
    /// (Document ID, Score, Document Length)
    pub list: Vec<(K, f64, u64)>, // (key, score, document length)
}

impl<K> Hits<K> {
    pub fn new(vec: Vec<(K, f64, u64)>) -> Self {
        Hits { list: vec }
    }

    pub fn sort_by_score(&mut self) -> &mut Self {
    // NaN を除外 (必要なら末尾へ送る運用も可)
    self.list.retain(|(_, s, _)| !s.is_nan());
    // total_cmp で反射律/推移律を満たす全順序 (NaN 排除済みなので安全)
    self.list.sort_by(|a, b| b.1.total_cmp(&a.1));
    self
    }

    pub fn sort_by_score_rev(&mut self) -> &mut Self{
        // NaN を除外 (必要なら末尾へ送る運用も可)
        self.list.retain(|(_, s, _)| !s.is_nan());
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
            for (key, score, doc_len) in &self.list {
                writeln!(f, "    {:?}: {:.6} (len: {})", key, score, doc_len)?;
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
{
    /// 内積によるスコアリング
    /// scoring by dot product
    fn scoring_dot(&self, freq: TokenFrequency) -> Vec<(K, f64, u64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);

        let doc_scores: Vec<(K, f64, u64)> = self.documents.iter().map(|doc| {(
            doc.key.clone(),
            tf.raw_iter().map(|(idx, val)| {
                let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                let tf2: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                let tf1: f64 = val.denormalize(tf_denormalize_num);
                // 内積の計算
                // dot calculation
                tf1 * tf2 * (idf * idf)
            }).sum::<f64>(),
            doc.token_sum
        )}).collect();
        doc_scores
    }

    /// コサイン類似度によるスコアリング
    /// scoring by cosine similarity
    /// cosθ = A・B / (|A||B|)
    fn scoring_cosine(&self, freq: TokenFrequency) -> Vec<(K, f64, u64)> {
        let (tf_1, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let doc_scores: Vec<(K, f64, u64)> = self.documents.iter().map(|doc| {
            let tf_1 = tf_1.raw_iter();
            let tf_2 = doc.tf_vec.raw_iter();
            let mut a_it = tf_1.fuse();
            let mut b_it = tf_2.fuse();
            let mut a_next = a_it.next();
            let mut b_next = b_it.next();
            let mut norm_a = 0_f64;
            let mut norm_b = 0_f64;
            let mut dot = 0_f64;
            while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
                match ia.cmp(&ib) {
                    Ordering::Equal => {
                        let idf = self.idf.idf_vec.get(ia).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                        norm_a += (va.denormalize(tf_denormalize_num) * idf).pow(2);
                        norm_b += (vb.denormalize(doc.denormalize_num) * idf).pow(2);
                        dot += va.denormalize(tf_denormalize_num) * vb.denormalize(doc.denormalize_num) * (idf * idf);
                        a_next = a_it.next();
                        b_next = b_it.next();
                    }
                    Ordering::Less => {
                        norm_a += (va.denormalize(tf_denormalize_num) * self.idf.idf_vec.get(ia).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num)).pow(2);
                        a_next = a_it.next();
                    }
                    Ordering::Greater => {
                        norm_b += (vb.denormalize(doc.denormalize_num) * self.idf.idf_vec.get(ib).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num)).pow(2);
                        b_next = b_it.next();
                    }
                }
            }
            let norm_a = norm_a.sqrt();
            let norm_b = norm_b.sqrt();
            let score = dot / (norm_a * norm_b);
            (doc.key.clone(), score, doc.token_sum)
        }).collect();
        doc_scores
    }

    /// BM25によるスコアリング
    /// scoring by BM25
    fn scoring_bm25(&self, freq: TokenFrequency, k1: f64, b: f64) -> Vec<(K, f64, u64)> {
        let (tf, _tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let k1_p = k1 + 1.0;
        // ドキュメントの平均長さ
        // average document length
        let avg_l = self.documents.iter().map(|doc| doc.token_sum as f64).sum::<f64>() / self.documents.len() as f64;

        let doc_scores: Vec<(K, f64, u64)> = self.documents.iter().map(|doc| {(
            doc.key.clone(),
            tf.raw_iter().map(|(idx, _val)| {
                let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                let tf: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                // BM25のスコア計算式
                // BM25 scoring formula
                idf * ((tf * k1_p) / (tf + k1 * (1.0 - b + (b * (doc.token_sum as f64 / avg_l)))))
            }).sum::<f64>(),
            doc.token_sum
        )}).collect();
        doc_scores
    }
} 