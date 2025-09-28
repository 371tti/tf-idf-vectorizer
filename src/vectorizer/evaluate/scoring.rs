use std::{cmp::Ordering, fmt::Debug};

use num::{pow::Pow, Num};
use rayon::prelude::*;

use crate::{utils::{math::vector::ZeroSpVecTrait, normalizer::DeNormalizer}, vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer}};

/// Enum for similarity algorithms used in search queries
pub enum SimilarityAlgorithm {
    /// Dot product similarity
    /// Considers both direction and magnitude
    Dot,
    /// Cosine similarity
    /// Considers only direction
    CosineSimilarity,
    /// BM25 similarity
    /// Considers document length
    /// param k1: Controls term frequency saturation
    /// param b: Controls document length normalization
    BM25(f64, f64), // (k1, b)
    /// BM25+ similarity
    /// Improves BM25 by adding a delta to term frequency
    /// param k1: Controls term frequency saturation
    /// param b: Controls document length normalization
    /// param delta: Small constant added to term frequency to avoid zero scores
    BM25plus(f64, f64, f64), // (k1, b, delta)
    /// BM25L similarity
    /// Considers document length with lower bound adjustment
    /// param k1: Controls term frequency saturation
    /// param b: Controls document length normalization
    BM25L(f64, f64), // (k1, b)
    /// Combined BM25 and Cosine Similarity
    /// param k1: Controls term frequency saturation for BM25
    /// param b: Controls document length normalization for BM25
    BM25CosineFilter(f64, f64), // (k1, b)
    /// Combined BM25 and Cosine Similarity with normalized linear combination
    /// param k1: Controls term frequency saturation for BM25
    /// param b: Controls document length normalization for BM25
    /// param alpha: Controls the balance between BM25 and Cosine Similarity (0.0 - 1.0)
    BM25CosineNormalizedLinearCombination(f64, f64, f64), // (k1, b, alpha)
    /// Combined BM25 pseudo-relevance feedback from top documents and Cosine Similarity
    /// param k1: Controls term frequency saturation for BM25
    /// param b: Controls document length normalization for BM25
    /// param top_n: Number of top documents to consider for pseudo-relevance feedback
    /// param alpha: balance of original query and pseudo-relevance feedback (0.0 is ignore original query, 1.0 is full weight to original query)
    BM25PrfCosineSimilarity(f64, f64, usize, f64), // (k1, b, top_n, alpha)
}

/// Structure to store search results
pub struct Hits<K> 
{
    /// (Document ID, Score, Document Length)
    pub list: Vec<(K, f64, u64)>, // (key, score, document length)
}

impl<K> Hits<K> {
    /// Create a new Hits instance
    pub fn new(vec: Vec<(K, f64, u64)>) -> Self {
        Hits { list: vec }
    }

    /// Sort results by descending score
    pub fn sort_by_score(&mut self) -> &mut Self {
        // Remove NaN scores
        self.list.retain(|(_, s, _)| !s.is_nan());
        // Sort by score descending
        self.list.sort_by(|a, b| b.1.total_cmp(&a.1));
        self
    }

    /// Sort results by ascending score
    pub fn sort_by_score_rev(&mut self) -> &mut Self{
        // Remove NaN scores
        self.list.retain(|(_, s, _)| !s.is_nan());
        // Sort by score ascending
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

impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    K: Clone + Sync + Send + PartialEq,
    N: Num + Copy + Into<f64> + DeNormalizer + Send + Sync,
    E: TFIDFEngine<N>,
{
    /// Calculate similarity scores based on query token frequency
    /// Uses the specified similarity algorithm
    /// Calls update_idf() to use the latest IDF vector
    pub fn similarity(&mut self, freq: &TokenFrequency, algorithm: &SimilarityAlgorithm) -> Hits<K> {
        self.update_idf();
        self.similarity_uncheck_idf(freq, algorithm)
    }

    /// Calculate similarity scores based on query token frequency
    /// Uses the specified similarity algorithm
    /// Does not check the IDF vector (can be called with immutable reference)
    /// Call update_idf() manually if needed
    pub fn similarity_uncheck_idf(&self, freq: &TokenFrequency, algorithm: &SimilarityAlgorithm) -> Hits<K> {
        // helper: safe min-max 正規化 (全要素同値なら 0.5 固定を返し順位保持)
        #[inline]
        fn normalize_values(mut vals: Vec<f64>) -> (Vec<f64>, f64, f64) {
            // min / max 計算 (NaN/Inf は除外)
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &v in &vals {
                if !v.is_finite() { continue; }
                if v < min_v { min_v = v; }
                if v > max_v { max_v = v; }
            }
            if !min_v.is_finite() { // 全て非有限
                for v in &mut vals { *v = 0.5; }
                return (vals, f64::NAN, f64::NAN);
            }
            let range = max_v - min_v;
            // 相対トレランス: 巨大値スケールでも安定
            let tol = (max_v.abs().max(min_v.abs())) * 1e-12 + 1e-15;
            if !range.is_finite() || range <= tol {
                // ほぼ一定 → 0.5 だがタイブレーク用にわずかに減少させ順位安定 (最大のみ 0.5)
                let n = vals.len();
                if n == 0 { return (vals, min_v, max_v); }
                for (i,v) in vals.iter_mut().enumerate() {
                    // 1e-15 * i で単調減少 (表示丸めで消えるがソートでは差異)
                    *v = 0.5 - (i as f64)*1e-15;
                }
                return (vals, min_v, max_v);
            }
            let inv = 1.0 / range;
            for v in &mut vals {
                if v.is_finite() {
                    *v = (*v - min_v) * inv; // [0,1]
                } else {
                    *v = 0.0; // 非有限は最下位近くへ
                }
            }
            // 最大値タイ (==1.0) が複数なら僅差オフセット付与してつぶれ回避
            let max_indices: Vec<usize> = vals.iter().enumerate().filter(|(_,v)| **v >= 1.0).map(|(i,_)| i).collect();
            if max_indices.len() > 1 {
                // 安全のため最大を厳密 1.0, 以降 1.0 - ε*i
                for (rank,&idx) in max_indices.iter().enumerate() {
                    vals[idx] = 1.0 - (rank as f64)*1e-12; // rank=0 は 1.0
                }
            }
            (vals, min_v, max_v)
        }

        let result = match algorithm {
            SimilarityAlgorithm::Dot => self.scoring_dot(&freq),
            SimilarityAlgorithm::CosineSimilarity => self.scoring_cosine(&freq),
            SimilarityAlgorithm::BM25(k1, b) => self.scoring_bm25(&freq, *k1, *b),
            SimilarityAlgorithm::BM25plus(k1, b, delta) => self.scoring_bm25plus(&freq, *k1, *b, *delta),
            SimilarityAlgorithm::BM25L(k1, b) => self.scoring_bm25l(&freq, *k1, *b),
            SimilarityAlgorithm::BM25CosineFilter(k1, b) => {
                // 直接乗算だと idf の累乗 (≈idf^3) になり偏りが強すぎるため双方を min-max 正規化後に幾何平均に変更
                let bm25 = self.scoring_bm25(&freq, *k1, *b);
                let cosine = self.scoring_cosine(&freq); // こちらは idf^2 を内包している
                let bm25_vals: Vec<f64> = bm25.iter().map(|(_, s, _)| *s).collect();
                let cosine_vals: Vec<f64> = cosine.iter().map(|(_, s, _)| *s).collect();
                let (bm25_norm, _, _) = normalize_values(bm25_vals);
                let (cosine_norm, _, _) = normalize_values(cosine_vals);
                bm25.into_iter().zip(cosine_norm.into_iter()).zip(bm25_norm.into_iter())
                    .map(|(((k, _s, l), c_n), b_n)| (k, (b_n * c_n).sqrt(), l)) // 幾何平均
                    .collect()
            },
            SimilarityAlgorithm::BM25CosineNormalizedLinearCombination(k1, b, alpha) => {
                let alpha = alpha.clamp(0.0, 1.0);
                let bm25 = self.scoring_bm25(&freq, *k1, *b);
                let cosine = self.scoring_cosine(&freq);
                let bm25_vals: Vec<f64> = bm25.iter().map(|(_, s, _)| *s).collect();
                let cosine_vals: Vec<f64> = cosine.iter().map(|(_, s, _)| *s).collect();
                let (bm25_norm, _, _) = normalize_values(bm25_vals);
                let (cosine_norm, _, _) = normalize_values(cosine_vals);
                bm25.into_iter().zip(bm25_norm.into_iter()).zip(cosine_norm.into_iter())
                    .map(|(((k, _s, l), b_n), c_n)| (k, alpha * b_n + (1.0 - alpha) * c_n, l))
                    .collect()
            },
            SimilarityAlgorithm::BM25PrfCosineSimilarity(k1, b, top_n, alpha) => {
                let mut cosine_scores = self.scoring_cosine(&freq);
                cosine_scores.retain(|(_, s, _)| !s.is_nan());
                cosine_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
                let mut prf_freq = TokenFrequency::new();
                cosine_scores.iter().take(*top_n).for_each(|(doc_key, _, _)| {
                    if let Some(cos_freq) = self.get_tf_into_token_freq(doc_key) {
                        prf_freq.add_tokens_from_freq(&cos_freq);
                    }
                });
                let original_bm25_scores = self.scoring_bm25(&freq, *k1, *b);
                let prf_bm25_scores = self.scoring_bm25(&prf_freq, *k1, *b);
                let alpha = alpha.clamp(0.0, 1.0);
                let orig_vals: Vec<f64> = original_bm25_scores.iter().map(|(_, s, _)| *s).collect();
                let prf_vals: Vec<f64> = prf_bm25_scores.iter().map(|(_, s, _)| *s).collect();
                let (orig_norm, _, _) = normalize_values(orig_vals);
                let (prf_norm, _, _) = normalize_values(prf_vals);
                original_bm25_scores.into_iter().zip(prf_norm.into_iter()).zip(orig_norm.into_iter())
                    .map(|(((k, _s, l), prf_n), orig_n)| (k, alpha * orig_n + (1.0 - alpha) * prf_n, l))
                    .collect()
            },
        };

        Hits { list: result }
    }
}

/// High-level search implementations
impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    K: Clone + Send + Sync + PartialEq,
    N: Num + Copy + Into<f64> + DeNormalizer + Send + Sync,
    E: TFIDFEngine<N>,
{
    /// Scoring by dot product
    fn scoring_dot(&self, freq: &TokenFrequency) -> Vec<(K, f64, u64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);

        let doc_scores: Vec<(K, f64, u64)> = self.documents.par_iter().map(|doc| {(
            doc.key.clone(),
            tf.raw_iter().map(|(idx, val)| {
                let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                let tf2: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                let tf1: f64 = val.denormalize(tf_denormalize_num);
                // Dot product calculation
                tf1 * tf2 * (idf * idf)
            }).sum::<f64>(),
            doc.token_sum
        )}).collect();
        doc_scores
    }

    /// Scoring by cosine similarity
    /// cosθ = A・B / (|A||B|)
    fn scoring_cosine(&self, freq: &TokenFrequency) -> Vec<(K, f64, u64)> {
        let (tf_1, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let doc_scores: Vec<(K, f64, u64)> = self.documents.par_iter().map(|doc| {
            let tf_1 = tf_1.raw_iter();
            let tf_2 = doc.tf_vec.raw_iter();
            let mut a_it = tf_1.fuse();
            let mut b_it = tf_2.fuse();
            let mut a_next = a_it.next();
            let mut b_next = b_it.next();
            let mut norm_a = 0_f64;
            let mut norm_b = 0_f64;
            let mut dot = 0_f64;
            // helper closure to fetch idf weight (denormalized). Missing indices get zero.
            let idf_w = |i: usize| -> f64 {
                self.idf
                    .idf_vec
                    .get(i)
                    .copied()
                    .unwrap_or(N::zero())
                    .denormalize(self.idf.denormalize_num)
            };
            while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
                match ia.cmp(&ib) {
                    Ordering::Equal => {
                        let idf = idf_w(ia);
                        norm_a += (va.denormalize(tf_denormalize_num) * idf).pow(2);
                        norm_b += (vb.denormalize(doc.denormalize_num) * idf).pow(2);
                        dot += va.denormalize(tf_denormalize_num) * vb.denormalize(doc.denormalize_num) * (idf * idf);
                        a_next = a_it.next();
                        b_next = b_it.next();
                    }
                    Ordering::Less => {
                        let idf = idf_w(ia);
                        norm_a += (va.denormalize(tf_denormalize_num) * idf).pow(2);
                        a_next = a_it.next();
                    }
                    Ordering::Greater => {
                        let idf = idf_w(ib);
                        norm_b += (vb.denormalize(doc.denormalize_num) * idf).pow(2);
                        b_next = b_it.next();
                    }
                }
            }
            // Remaining terms on the query side (a)
            while let Some((ia, va)) = a_next {
                let idf = idf_w(ia);
                norm_a += (va.denormalize(tf_denormalize_num) * idf).pow(2);
                a_next = a_it.next();
            }
            // Remaining terms on the document side (b)
            while let Some((ib, vb)) = b_next {
                let idf = idf_w(ib);
                norm_b += (vb.denormalize(doc.denormalize_num) * idf).pow(2);
                b_next = b_it.next();
            }
            let norm_a = norm_a.sqrt();
            let norm_b = norm_b.sqrt();
            // Zero division safety with f64::EPSILON
            let score = dot / (norm_a * norm_b + f64::EPSILON);
            (doc.key.clone(), score, doc.token_sum)
        }).collect();
        doc_scores
    }

    /// Scoring by BM25
    fn scoring_bm25(&self, freq: &TokenFrequency, k1: f64, b: f64) -> Vec<(K, f64, u64)> {
        let (tf, _tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let k1_p = k1 + 1.0;
        // Average document length
        let avg_l = self.documents.iter().map(|doc| doc.token_sum as f64).sum::<f64>() / self.documents.len() as f64;
        let rev_avg_l = 1.0 / avg_l;

        let doc_scores: Vec<(K, f64, u64)> = self.documents.par_iter().map(|doc| {(
            doc.key.clone(),
            {
                let len_p = doc.token_sum as f64 * rev_avg_l;
                tf.raw_iter().map(|(idx, _val)| {
                    let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                    let tf: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                    // BM25 scoring formula
                    idf * ((tf * k1_p) / (tf + k1 * (1.0 - b + (b * len_p))))
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }

    /// Scoring by BM25+
    fn scoring_bm25plus(&self, freq: &TokenFrequency, k1: f64, b: f64, delta: f64) -> Vec<(K, f64, u64)> {
        let (tf, _tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        // Average document length
        let avg_l = self.documents.iter().map(|doc| doc.token_sum as f64).sum::<f64>() / self.documents.len() as f64;
        let rev_avg_l = 1.0 / avg_l;
        let doc_scores: Vec<(K, f64, u64)> = self.documents.par_iter().map(|doc| {(
            doc.key.clone(),
            {
                let len_p = doc.token_sum as f64 * rev_avg_l;
                tf.raw_iter().map(|(idx, _val)| {
                    let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                    let tf: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                    // BM25+ scoring formula
                    let denom = tf + k1 * (1.0 - b + (b * len_p));
                    idf * (((tf + delta) * (k1 + 1.0)) / denom)
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }

    /// Scoring by BM25L
    fn scoring_bm25l(&self, freq: &TokenFrequency, k1: f64, b: f64) -> Vec<(K, f64, u64)> {
        let (tf, _tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let k1_p = k1 + 1.0;
        // Average document length
        let avg_l = self.documents.iter().map(|doc| doc.token_sum as f64).sum::<f64>() / self.documents.len() as f64;
        let rev_avg_l = 1.0 / avg_l;

        let doc_scores: Vec<(K, f64, u64)> = self.documents.par_iter().map(|doc| {(
            doc.key.clone(),
            {
                let len_p = doc.token_sum as f64 * rev_avg_l;
                tf.raw_iter().map(|(idx, _val)| {
                    let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                    let tf: f64 = doc.tf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                    // BM25L: normalize term frequency by length factor then apply BM25 form
                    // tf' = tf / ( (1 - b) + b * (len/avg_len) )
                    let norm = 1.0 - b + b * len_p;
                    let tf_norm = if norm > 0.0 { tf / norm } else { tf }; // safety
                    idf * ((tf_norm * k1_p) / (tf_norm + k1))
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }
}