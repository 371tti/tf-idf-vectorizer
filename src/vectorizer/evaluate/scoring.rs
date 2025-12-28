use std::{borrow::Borrow, cmp::Ordering, fmt::{Debug, Display}, hash::Hash, ops::Deref};

use num_traits::{pow::Pow, Num};

use crate::{Query, TFIDFEngine, TFIDFVectorizer, TermFrequency, utils::{datastruct::vector::{TFVector, TFVectorTrait}}, vectorizer::KeyRc};

/// Enum for similarity algorithms used in search queries
#[derive(Clone)]
pub enum SimilarityAlgorithm {
    /// Contains
    /// Checks if documents contain the query terms
    Contains,
    /// Dot product similarity
    /// Considers both direction and magnitude
    Dot,
    /// Cosine similarity
    /// Considers only direction
    CosineSimilarity,
    /// BM25-Like similarity
    /// Considers document length
    /// param k1: Controls term frequency saturation
    /// param b: Controls document length normalization
    BM25(f64, f64), // (k1, b)
}

/// Structure to store search results
#[derive(Clone, Default)]
pub struct Hits<K> 
{
    /// (Document ID, Score, Document Length)
    pub list: Vec<HitEntry<K>>, // (key, score, document term sum)
}

#[derive(Clone, Debug)]
pub struct HitEntry<K> {
    pub key: K,
    pub score: f64,
    pub doc_len: u32,
}

impl<K> Display for HitEntry<K> 
where
    K: Debug,
{
    /// "score: {:.6}\tdoc_len: {}\tkey: {:?}"
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = f.precision().unwrap_or(6);
        write!(f, "score: {:.*}\tdoc_len: {}\tkey: {:?}", precision, self.score, self.doc_len, self.key)
    }
    
}

impl<K> Hits<K> {
    /// Count the number of non-zero and non-NaN scores in the results
    pub fn count_non_zero_scores(&self) -> usize {
        self.list.iter().filter(|&s| s.score != 0.0 && !s.score.is_nan()).count()
    }

    /// Count the number of zero scores in the results
    pub fn count_zero_scores(&self) -> usize {
        self.list.iter().filter(|&s| s.score == 0.0).count()
    }

    /// Count the number of NaN scores in the results
    pub fn count_nan_scores(&self) -> usize {
        self.list.iter().filter(|&s| s.score.is_nan()).count()
    }

    /// Sort results by score in ascending order
    /// Removes NaN scores
    pub fn sort_by_score_asc(&mut self) -> &mut Self{
        // Remove NaN scores
        self.list.retain(|s| !s.score.is_nan());
        // Sort by score ascending
        self.list.sort_by(|a, b| a.score.total_cmp(&b.score));
        self
    }

    /// Sort results by score in descending order
    /// Removes NaN scores
    pub fn sort_by_score_desc(&mut self) -> &mut Self{
        // Remove NaN scores
        self.list.retain(|s| !s.score.is_nan());
        // Sort by score descending
        self.list.sort_by(|a, b| b.score.total_cmp(&a.score));
        self
    }

    pub fn sort_by(&mut self, cmp: impl FnMut(&HitEntry<K>, &HitEntry<K>) -> Ordering) -> &mut Self {
        self.list.sort_by(cmp);
        self
    }

    /// Keep only the top k results
    /// use after sorting
    pub fn top_k(&mut self, k: usize) -> &mut Self {
        if self.list.len() > k {
            self.list.truncate(k);
        }
        self
    }
}

/// Special sorting implementations
impl<K> Hits<K> 
{
    /// タイトルに特定の文字列を含むものを優先的に上位に持ってくるソート
    /// 辞書順で
    /// 1. タイトルにQが含まれる
    /// 2. スコアが高い
    pub fn sort_by_title_score_desc<Q>(&mut self, contains_str: &Q) -> &mut Self 
    where
        K: Borrow<str>,
        Q: ?Sized + AsRef<str>,
    {
        self.list.retain(|s| !s.score.is_nan());
        self.list.sort_unstable_by(|a, b| {
            let a_contains = a.key.borrow().contains(contains_str.as_ref());
            let b_contains = b.key.borrow().contains(contains_str.as_ref());
            match (a_contains, b_contains) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => b.score.total_cmp(&a.score),
            }
        });
        self
    }
}

impl<K> FromIterator<HitEntry<K>> for Hits<K> {
    fn from_iter<T: IntoIterator<Item = HitEntry<K>>>(iter: T) -> Self {
        let list: Vec<HitEntry<K>> = iter.into_iter().collect();
        Hits { list }
    }
}

impl<K> Display for Hits<K> 
where
    K: Debug,
{
    /// Display the hits in a readable format
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for hit in &self.list {
            writeln!(f, "{}", hit)?;
        }
        Ok(())
    }
}

impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    K: Clone + Sync + Send + PartialEq + Eq + Hash,
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
{
    /// High-level similarity search interface
    pub fn similarity(&mut self, algorithm: &SimilarityAlgorithm, freq: &TermFrequency, query: Option<&Query>) -> Hits<K> {
        self.update_idf();
        self.similarity_uncheck_idf(algorithm, freq, query)
    }

    /// High-level similarity search interface without IDF update check
    pub fn similarity_uncheck_idf(&self, algorithm: &SimilarityAlgorithm, freq: &TermFrequency, filter_query: Option<&Query>) -> Hits<K> {
        let binding = Query::from_freq_or(freq);
        let doc_iter = self.optimized_iter(filter_query.unwrap_or(&binding).build(&self.term_dim_rev_index, &self.documents.as_index_set()));
        match algorithm {
            SimilarityAlgorithm::Contains => self.contains_docs(freq),
            SimilarityAlgorithm::Dot => self.scoring_dot(freq, doc_iter),
            SimilarityAlgorithm::CosineSimilarity => self.scoring_cosine(freq, doc_iter),
            SimilarityAlgorithm::BM25(k1, b) => self.scoring_bm25(freq, *k1, *b, doc_iter),
        }
    }

    /// High-level search interface
    pub fn search(&mut self, algorithm: &SimilarityAlgorithm, query: Query) -> Hits<K> {
        self.update_idf();
        self.search_uncheck_idf(algorithm, query)
    }

    /// High-level search interface without IDF update check
    pub fn search_uncheck_idf(&self, algorithm: &SimilarityAlgorithm, query: Query) -> Hits<K> {
        let freq = TermFrequency::from(query.get_all_terms().as_slice());
        let doc_iter =  self.optimized_iter(query.build(&self.term_dim_rev_index, &self.documents.as_index_set()));
        match algorithm {
            SimilarityAlgorithm::Contains => self.contains_docs(&freq),
            SimilarityAlgorithm::Dot => self.scoring_dot(&freq, doc_iter),
            SimilarityAlgorithm::CosineSimilarity => self.scoring_cosine(&freq, doc_iter),
            SimilarityAlgorithm::BM25(k1, b) => self.scoring_bm25(&freq, *k1, *b, doc_iter),
        }
    }
}

impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    K: Clone + Send + Sync + PartialEq + Eq + Hash,
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
{
    fn optimized_iter<'a>(&'a self, filter: Vec<usize>) -> OptimizedDocIter<'a, K, N, E> {
        OptimizedDocIter {
            contains_indices: filter,
            vectorizer: self,
            current_opt_idx: 0,
        }
    }
}

pub struct OptimizedDocIter<'a, K, N, E>
where
    K: Clone + Send + Sync + PartialEq + Eq + Hash,
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
{
    contains_indices: Vec<usize>,
    vectorizer: &'a TFIDFVectorizer<N, K, E>,
    current_opt_idx: usize,
}

impl<'a, K, N, E> Iterator for OptimizedDocIter<'a, K, N, E>
where
    K: Clone + Send + Sync + PartialEq + Eq + Hash,
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
{
    type Item = (&'a KeyRc<K>, &'a TFVector<N>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_opt_idx >= self.contains_indices.len() {
            return None;
        }
        let doc_idx = self.contains_indices[self.current_opt_idx];
        self.current_opt_idx += 1;
        self.vectorizer.documents.get_key_value_with_index(doc_idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.contains_indices.len() - self.current_opt_idx;
        (remaining, Some(remaining))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.current_opt_idx += n;
        self.next()
    }
}

/// High-level search implementations
impl<'a, N, K, E > TFIDFVectorizer<N, K, E>
where
    K: Clone + Send + Sync + PartialEq + Eq + Hash + 'a,
    N: Num + Copy + Into<f64> + Send + Sync + 'a,
    E: TFIDFEngine<N> + Send + Sync,
{
    /// Contains document indices that have at least one term in freq
    fn contains_docs(&self, freq: &TermFrequency) -> Hits<K> {
        let mut doc_indices: Vec<usize> = freq.term_set_ref_str().iter().flat_map(|&term| {
            self.term_dim_rev_index.get(term).map(|keys| {
                keys.iter().filter_map(|key| {
                    self.documents.get_index(key)
                }).collect::<Vec<usize>>()
            }).unwrap_or_else(Vec::new)
        }).collect();
        doc_indices.sort_unstable();
        doc_indices.dedup();
        doc_indices.iter().map(|&idx| {
            let (key, doc_tf_vec) = self.documents.get_key_value_with_index(idx).unwrap();
            HitEntry {
                key: key.deref().clone(),
                score: 1.0,
                doc_len: doc_tf_vec.term_sum(),
            }
        }).collect()
    }

    /// Scoring by dot product
    fn scoring_dot(&self, freq: &TermFrequency, doc_iter: impl Iterator<Item = (&'a KeyRc<K>, &'a TFVector<N>)>) -> Hits<K> {
        let query_tf_vec = E::tf_vec(&freq, self.term_dim_rev_index.as_index_set());

        let doc_scores = doc_iter.map(|(key, doc_tf_vec)| 
            HitEntry {
                key: key.deref().clone(),
                score: {
                    let mut cut_down = 0;
                    query_tf_vec.raw_iter().map(|(idx, val)| {
                        let idf = self.idf_cache.idf_vec.get(idx as usize).copied().unwrap_or(0.0);
                        let tf2 = E::tf_denorm(doc_tf_vec.get_power_jump(idx, &mut cut_down).unwrap_or(N::zero()));
                        let tf1 = E::tf_denorm(val);
                        // Dot product calculation
                        tf1 as f64 * tf2 as f64 * (idf as f64 * idf as f64)
                    }).sum::<_>()
                },
                doc_len: doc_tf_vec.term_sum()
            }
        ).collect();
        doc_scores
    }

    /// Scoring by cosine similarity
    /// cosθ = A・B / (|A||B|)
    fn scoring_cosine(&self, freq: &TermFrequency, doc_iter: impl Iterator<Item = (&'a KeyRc<K>, &'a TFVector<N>)>) -> Hits<K> {
        let query_tf_vec= E::tf_vec(&freq, self.term_dim_rev_index.as_index_set());

        let doc_scores = doc_iter.map(|(key, doc_tf_vec)| {
            let tf_1 = query_tf_vec.raw_iter();
            let tf_2 = doc_tf_vec.raw_iter();
            let mut a_it = tf_1.fuse();
            let mut b_it = tf_2.fuse();
            let mut a_next = a_it.next();
            let mut b_next = b_it.next();
            let mut norm_a = 0_f32;
            let mut norm_b = 0_f32;
            let mut dot = 0_f32;
            // helper closure to fetch idf weight (denormalized). Missing indices get zero.
            let idf_w = |i: u32| -> f32 {
                self.idf_cache
                    .idf_vec
                    .get(i as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            while let (Some((ia, va)), Some((ib, vb))) = (a_next, b_next) {
                match ia.cmp(&ib) {
                    Ordering::Equal => {
                        let idf = idf_w(ia);
                        norm_a += (E::tf_denorm(va) as f32 * idf).pow(2);
                        norm_b += (E::tf_denorm(vb) as f32 * idf).pow(2);
                        dot += E::tf_denorm(va) as f32 * E::tf_denorm(vb) as f32 * (idf * idf);
                        a_next = a_it.next();
                        b_next = b_it.next();
                    }
                    Ordering::Less => {
                        let idf = idf_w(ia);
                        norm_a += (E::tf_denorm(va) as f32 * idf).pow(2);
                        a_next = a_it.next();
                    }
                    Ordering::Greater => {
                        let idf = idf_w(ib);
                        norm_b += (E::tf_denorm(vb) as f32 * idf).pow(2);
                        b_next = b_it.next();
                    }
                }
            }
            // Remaining terms on the query side (a)
            while let Some((ia, va)) = a_next {
                let idf = idf_w(ia);
                norm_a += (E::tf_denorm(va) as f32 * idf).pow(2);
                a_next = a_it.next();
            }
            // Remaining terms on the document side (b)
            while let Some((ib, vb)) = b_next {
                let idf = idf_w(ib);
                norm_b += (E::tf_denorm(vb) as f32 * idf).pow(2);
                b_next = b_it.next();
            }
            let norm_a = norm_a.sqrt();
            let norm_b = norm_b.sqrt();
            // Zero division safety with f64::EPSILON
            let score = dot as f64 / (norm_a as f64 * norm_b as f64 + f64::EPSILON);
            HitEntry {
                key: key.deref().clone(),
                score,
                doc_len: doc_tf_vec.term_sum(),
            }
        }).collect();
        doc_scores
    }

    /// Scoring by BM25-Like
    fn scoring_bm25(&self, freq: &TermFrequency, k1: f64, b: f64, doc_iter: impl Iterator<Item = (&'a KeyRc<K>, &'a TFVector<N>)>) -> Hits<K> {
        let query_tf_vec = E::tf_vec(&freq, self.term_dim_rev_index.as_index_set());

        let k1_p = k1 + 1.0;
        // Average document length in the vectorizer
        let avg_l = self.documents.iter().map(|(_k, doc_tf_vec)| doc_tf_vec.term_sum() as f64).sum::<f64>() / self.documents.len() as f64;
        let rev_avg_l = 1.0 / avg_l;

        let doc_scores = doc_iter.map(|(key, doc_tf_vec)| 
            HitEntry {
                key: key.deref().clone(),
                score: {
                    let len_p = doc_tf_vec.term_sum() as f64 * rev_avg_l;
                    let mut cut_down = 0;
                    query_tf_vec.raw_iter().map(|(idx, _qtf)| {
                        let idf = self.idf_cache.idf_vec.get(idx as usize).copied().unwrap_or(0.0).ln();
                        let dtf = E::tf_denorm(doc_tf_vec.get_power_jump(idx, &mut cut_down).unwrap_or(N::zero()));
                        // BM25 scoring formula
                        idf as f64 * ((dtf as f64 * k1_p) / (dtf as f64 + k1 * (1.0 - b + (b * len_p))))
                    }).sum::<_>()
                },
                doc_len: doc_tf_vec.term_sum()
            }
        ).collect();
        doc_scores
    }
}