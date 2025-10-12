use std::{cmp::Ordering, fmt::Debug};

use num::{pow::Pow, Num};

use crate::{utils::{math::vector::ZeroSpVecTrait, normalizer::DeNormalizer}, vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer}};

/// Enum for similarity algorithms used in search queries
#[derive(Clone)]
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
}

/// Structure to store search results
#[derive(Clone)]
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
    E: TFIDFEngine<N> + Send + Sync,
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
        };

        Hits { list: result }
    }
}

/// High-level search implementations
impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    K: Clone + Send + Sync + PartialEq,
    N: Num + Copy + Into<f64> + DeNormalizer + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
{
    /// Scoring by dot product
    fn scoring_dot(&self, freq: &TokenFrequency) -> Vec<(K, f64, u64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);

        let doc_scores: Vec<(K, f64, u64)> = self.documents.iter().map(|doc| {(
            doc.key.clone(),
            {
                let mut cut_down = 0;
                tf.raw_iter().map(|(idx, val)| {
                    let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                    let tf2 = doc.tf_vec.raw_get_with_cut_down(idx, cut_down).map(|v| {
                        cut_down = v.index + 1; // Update cut_down to skip processed indices
                        v.value
                    }).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                    let tf1: f64 = val.denormalize(tf_denormalize_num);
                    // Dot product calculation
                    tf1 * tf2 * (idf * idf)
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }

    /// Scoring by cosine similarity
    /// cosθ = A・B / (|A||B|)
    fn scoring_cosine(&self, freq: &TokenFrequency) -> Vec<(K, f64, u64)> {
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

        let doc_scores: Vec<(K, f64, u64)> = self.documents.iter().map(|doc| {(
            doc.key.clone(),
            {
                let len_p = doc.token_sum as f64 * rev_avg_l;
                let mut cut_down = 0;
                tf.raw_iter().map(|(idx, _val)| {
                    let idf: f64 = self.idf.idf_vec.get(idx).copied().unwrap_or(N::zero()).denormalize(self.idf.denormalize_num);
                    let tf: f64 = doc.tf_vec.raw_get_with_cut_down(idx, cut_down).map(|v| {
                        cut_down = v.index + 1; // Update cut_down to skip processed indices
                        v.value
                    }).copied().unwrap_or(N::zero()).denormalize(doc.denormalize_num);
                    // BM25 scoring formula
                    idf * ((tf * k1_p) / (tf + k1 * (1.0 - b + (b * len_p))))
                }).sum::<f64>()
            },
            doc.token_sum
        )}).collect();
        doc_scores
    }

    // fn search_bm25(&self, positive_freq: &TokenFrequency, negative_freq: &TokenFrequency, k1: f64, b: f64) -> Vec<(K, f64, u64)> {

    // }
}