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
        let result = match algorithm {
            SimilarityAlgorithm::Dot => self.scoring_dot(&freq),
            SimilarityAlgorithm::CosineSimilarity => self.scoring_cosine(&freq),
            SimilarityAlgorithm::BM25(k1, b) => self.scoring_bm25(&freq, *k1, *b),
            SimilarityAlgorithm::BM25plus(k1, b, delta) => self.scoring_bm25plus(&freq, *k1, *b, *delta),
            SimilarityAlgorithm::BM25L(k1, b) => self.scoring_bm25l(&freq, *k1, *b),
            SimilarityAlgorithm::BM25CosineFilter(k1, b) => {
                let mut bm25_scores = self.scoring_bm25(&freq, *k1, *b);
                let cosine_scores = self.scoring_cosine(&freq);
                // already sorted by document order, so we can multiply directly
                bm25_scores.iter_mut().zip(cosine_scores.iter()).for_each(|((_, bm25_score, _), (_, cosine_score, _))| {
                    *bm25_score *= *cosine_score;
                });
                bm25_scores
            },
            SimilarityAlgorithm::BM25CosineNormalizedLinearCombination(k1, b, alpha) => {
                let mut bm25_scores = self.scoring_bm25(&freq, *k1, *b);
                let cosine_scores = self.scoring_cosine(&freq);
                // Normalize scores to [0, 1]
                let (bm25_min, bm25_max) = bm25_scores.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &(_, score, _)| {
                    (min.min(score), max.max(score))
                });
                let (cosine_min, cosine_max) = cosine_scores.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &(_, score, _)| {
                    (min.min(score), max.max(score))
                });
                let alpha = alpha.clamp(0.0, 1.0);
                let rev_bm25_range = 1.0 / (bm25_max - bm25_min + f64::EPSILON);
                let rev_cosine_range = 1.0 / (cosine_max - cosine_min + f64::EPSILON);
                bm25_scores.iter_mut().zip(cosine_scores.iter()).for_each(|((_, bm25_score, _), (_, cosine_score, _))| {
                    let norm_bm25 = (*bm25_score - bm25_min) * rev_bm25_range;
                    let norm_cosine = (*cosine_score - cosine_min) * rev_cosine_range;
                    *bm25_score = alpha * norm_bm25 + (1.0 - alpha) * norm_cosine;
                });
                bm25_scores
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
                // Normalize scores to [0, 1]
                let (orig_min, orig_max) = original_bm25_scores.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &(_, score, _)| {
                    (min.min(score), max.max(score))
                });
                let (prf_min, prf_max) = prf_bm25_scores.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &(_, score, _)| {
                    (min.min(score), max.max(score))
                });
                let alpha = alpha.clamp(0.0, 1.0);
                let rev_orig_range = 1.0 / (orig_max - orig_min + f64::EPSILON);
                let rev_prf_range = 1.0 / (prf_max - prf_min + f64::EPSILON);
                let result: Vec<(K, f64, u64)> = original_bm25_scores.iter().zip(prf_bm25_scores.iter()).map(|((key1, orig_score, doc_len), (_, prf_score, _))| {
                    let norm_orig = (*orig_score - orig_min) * rev_orig_range;
                    let norm_prf = (*prf_score - prf_min) * rev_prf_range;
                    let combined_score = alpha * norm_orig + (1.0 - alpha) * norm_prf;
                    (key1.clone(), combined_score, *doc_len)
                }).collect();
                result
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
                    idf * ((tf + delta) / (tf + k1 * (1.0 - b + (b * len_p))))
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
                    let tf = tf * (1.0 - b + b * len_p);
                    // BM25 scoring formula
                    idf * ((tf * k1_p) / (tf + k1 * (1.0 - b + (b * len_p))))
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }
}