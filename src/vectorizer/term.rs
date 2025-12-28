use core::str;
use std::{collections::{HashMap, HashSet}, fmt::Debug};
use ahash::RandomState;
use serde::{Deserialize, Serialize};

use crate::Corpus;


/// TermFrequency struct
/// Manages the frequency of term occurrences.
/// Counts the number of times each term appears.
///
/// # Examples
/// ```
/// use crate::tf_idf_vectorizer::vectorizer::term::TermFrequency;
/// let mut term_freq = TermFrequency::new();
/// term_freq.add_term("term1");
/// term_freq.add_term("term2");
/// term_freq.add_term("term1");
///
/// assert_eq!(term_freq.term_count("term1"), 2);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TermFrequency {
    term_count: HashMap<String, u64, RandomState>,
    total_term_count: u64,
}

/// Implementation for adding and removing terms
impl TermFrequency {
    /// Create a new TermFrequency
    pub fn new() -> Self {
        TermFrequency {
            term_count: HashMap::with_hasher(RandomState::new()),
            total_term_count: 0,
        }
    }

    /// Add a term
    ///
    /// # Arguments
    /// * `term` - term to add
    #[inline]
    pub fn add_term(&mut self, term: &str) -> &mut Self {
        let count = self.term_count.entry(term.to_string()).or_insert(0);
        *count += 1;
        self.total_term_count += 1;
        self
    }

    /// Add multiple terms
    ///
    /// # Arguments
    /// * `terms` - Slice of terms to add
    #[inline]
    pub fn add_terms<T>(&mut self, terms: &[T]) -> &mut Self 
    where T: AsRef<str> 
    {
        for term in terms {
            let term_str = term.as_ref();
            self.add_term(term_str);
        }
        self
    }

    /// Subtract a term
    ///
    /// # Arguments
    /// * `term` - term to subtract
    #[inline]
    pub fn sub_term(&mut self, term: &str) -> &mut Self {
        if let Some(count) = self.term_count.get_mut(term) {
            if *count > 1 {
                *count -= 1;
                self.total_term_count -= 1;
            } else if *count == 1 {
                self.term_count.remove(term);
                self.total_term_count -= 1;
            }
        }
        self
    }

    /// Subtract multiple terms
    ///
    /// # Arguments
    /// * `terms` - Slice of terms to subtract
    #[inline]
    pub fn sub_terms<T>(&mut self, terms: &[T]) -> &mut Self 
    where T: AsRef<str>
    {
        for term in terms {
            let term_str = term.as_ref();
            self.sub_term(term_str);
        }
        self
    }

    /// Set the occurrence count for a term
    ///
    /// # Arguments
    /// * `term` - term
    /// * `count` - Occurrence count
    pub fn set_term_count(&mut self, term: &str, count: u64) -> &mut Self {
        if count == 0 {
            self.term_count.remove(term);
        } else {
            let current_count = self.term_count.entry(term.to_string()).or_insert(0);
            self.total_term_count += count - *current_count;
            *current_count = count;
        }
        self
    }

    /// Merge with another TermFrequency
    /// # Arguments
    /// * `other` - Another TermFrequency to merge with
    pub fn add_terms_from_freq(&mut self, other: &TermFrequency) -> &mut Self {
        for (term, &count) in &other.term_count {
            let entry = self.term_count.entry(term.clone()).or_insert(0);
            *entry += count;
            self.total_term_count += count;
        }
        self
    }

    /// Scale the term counts by a scalar
    /// # Arguments
    /// * `scalar` - Scalar to scale by
    pub fn scale(&mut self, scalar: f64) -> &mut Self {
        let mut total_count = 0;
        self.term_count.iter_mut().for_each(|(_, count)| {
            *count = ((*count as f64) * scalar).round() as u64;
            total_count += *count;
        });
        self.total_term_count = total_count;
        self
    }
}

impl<T> From<&[T]> for TermFrequency
where
    T: AsRef<str>,
{
    fn from(terms: &[T]) -> Self {
        let mut tf = TermFrequency::new();
        tf.add_terms(terms);
        tf
    }
}

impl From<Corpus> for TermFrequency {
    fn from(corpus: Corpus) -> Self {
        let mut tf = TermFrequency::new();
        for entry in corpus.term_counts.iter() {
            let term = entry.key();
            let count = *entry.value();
            tf.set_term_count(term, count);
        }
        tf
    }
}

/// Implementation for retrieving information from TermFrequency
impl TermFrequency {
    /// Get iterator over all terms and their counts
    /// 
    /// # Returns
    /// * `impl Iterator<Item=(&str, u64)>` - Iterator over terms and their counts
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item=(&str, u64)> {
        self.term_count.iter().map(|(term, &count)| {
            (term.as_str(), count)
        })
    }

    /// Get a vector of all terms and their counts
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of terms and their counts
    #[inline]
    pub fn term_count_vector(&self) -> Vec<(String, u64)> {
        self.term_count.iter().map(|(term, &count)| {
            (term.clone(), count)
        }).collect()
    }

    /// Get a vector of all terms and their counts (as &str)
    ///
    /// # Returns
    /// * `Vec<(&str, u64)>` - Vector of terms and their counts
    #[inline]
    pub fn term_count_vector_ref_str(&self) -> Vec<(&str, u64)> {
        self.term_count.iter().map(|(term, &count)| {
            (term.as_str(), count)
        }).collect()
    }

    /// Get a hashmap of all terms and their counts (as &str)
    ///
    /// # Returns
    /// * `HashMap<&str, u64>` - HashMap of terms and their counts
    #[inline]
    pub fn term_count_hashmap_ref_str(&self) -> HashMap<&str, u64, RandomState> {
        self.term_count.iter().map(|(term, &count)| {
            (term.as_str(), count)
        }).collect()
    }

    /// Get the total count of all terms
    ///
    /// # Returns
    /// * `u64` - Total term count
    #[inline]
    pub fn term_sum(&self) -> u64 {
        self.total_term_count
    }

    /// Get the occurrence count for a specific term
    ///
    /// # Arguments
    /// * `term` - term
    ///
    /// # Returns
    /// * `u64` - Occurrence count for the term
    #[inline]
    pub fn term_count(&self, term: &str) -> u64 {
        *self.term_count.get(term).unwrap_or(&0)
    }

    /// Get the most frequent terms
    /// If multiple terms have the same count, all are returned
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of most frequent terms and their counts
    #[inline]
    pub fn most_frequent_terms_vector(&self) -> Vec<(String, u64)> {
        if let Some(&max_count) = self.term_count.values().max() {
            self.term_count.iter()
                .filter(|&(_, &count)| count == max_count)
                .map(|(term, &count)| (term.clone(), count))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the count of the most frequent term
    ///
    /// # Returns
    /// * `u64` - Count of the most frequent term
    #[inline]
    pub fn most_frequent_term_count(&self) -> u64 {
        if let Some(&max_count) = self.term_count.values().max() {
            max_count
        } else {
            0
        }
    }

    /// Check if a term exists
    ///
    /// # Arguments
    /// * `term` - term
    ///
    /// # Returns
    /// * `bool` - true if the term exists, false otherwise
    #[inline]
    pub fn contains_term(&self, term: &str) -> bool {
        self.term_count.contains_key(term)
    }

    /// term_set_iter
    /// 
    /// # Returns
    /// * `impl Iterator<Item=&str>` - Iterator over the set of terms
    #[inline]
    pub fn term_set_iter(&self) -> impl Iterator<Item=&str> {
        self.term_count.keys().map(|s| s.as_str())
    }

    /// Get the set of terms
    ///
    /// # Returns
    /// * `Vec<String>` - Set of terms
    #[inline]
    pub fn term_set(&self) -> Vec<String> {
        self.term_count.keys().cloned().collect()
    }

    /// Get the set of terms (as &str)
    ///
    /// # Returns
    /// * `Vec<&str>` - Set of terms
    #[inline]
    pub fn term_set_ref_str(&self) -> Vec<&str> {
        self.term_count.keys().map(|s| s.as_str()).collect()
    }

    /// Get the set of terms as a HashSet
    ///
    /// # Returns
    /// * `HashSet<String>` - Set of terms
    #[inline]
    pub fn term_hashset(&self) -> HashSet<String, RandomState> {
        self.term_count.keys().cloned().collect()
    }

    /// Get the set of terms as a HashSet (as &str)
    ///
    /// # Returns
    /// * `HashSet<&str>` - Set of terms
    #[inline]
    pub fn term_hashset_ref_str(&self) -> HashSet<&str, RandomState> {
        self.term_count.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of unique terms
    ///
    /// # Returns
    /// * `usize` - Number of unique terms
    #[inline]
    pub fn term_num(&self) -> usize {
        self.term_count.len()
    }

    /// Remove stop terms
    ///
    /// # Arguments
    /// * `stop_terms` - Slice of stop terms to remove
    ///
    /// # Returns
    /// * `u64` - Total count of removed terms
    #[inline]
    pub fn remove_stop_terms(&mut self, stop_terms: &[&str]) -> u64{
        let mut removed_total_count: u64 = 0;
        for &stop_term in stop_terms {
            if let Some(count) = self.term_count.remove(stop_term) {
                removed_total_count += count as u64;
            }
        }
        self.total_term_count -= removed_total_count;
        removed_total_count
    }

    /// Remove terms by a condition
    ///
    /// # Arguments
    /// * `condition` - Closure to determine which terms to remove
    ///
    /// # Returns
    /// * `u64` - Total count of removed terms
    #[inline]
    pub fn remove_terms_by<F>(&mut self, condition: F) -> u64
    where
        F: Fn(&str, &u64) -> bool,
    {
        let mut removed_total_count: u64 = 0;
        self.term_count.retain(|term, count| {
            if condition(term, count) {
                removed_total_count += *count as u64;
                false
            } else {
                true
            }
        });
        self.total_term_count -= removed_total_count as u64;

        removed_total_count
    }

    /// Get a vector of terms sorted by frequency (descending)
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of terms sorted by frequency
    #[inline]
    pub fn sorted_frequency_vector(&self) -> Vec<(String, u64)> {
        let mut term_list: Vec<(String, u64)> = self.term_count
            .iter()
            .map(|(term, &count)| (term.clone(), count))
            .collect();

        term_list.sort_by(|a, b| b.1.cmp(&a.1));
        term_list
    }

    /// Get a vector of terms sorted by dictionary order (ascending)
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of terms sorted by dictionary order
    #[inline]
    pub fn sorted_dict_order_vector(&self) -> Vec<(String, u64)> {
        let mut term_list: Vec<(String, u64)> = self.term_count
            .iter()
            .map(|(term, &count)| (term.clone(), count))
            .collect();

        term_list.sort_by(|a, b| a.0.cmp(&b.0));
        term_list
    }

    /// Calculate the diversity of terms
    /// 1.0 indicates complete diversity, 0.0 indicates no diversity
    ///
    /// # Returns
    /// * `f64` - Diversity of terms
    #[inline]
    pub fn unique_term_ratio(&self) -> f64 {
        if self.total_term_count == 0 {
            return 0.0;
        }
        self.term_count.len() as f64 / self.total_term_count as f64
    }

    /// Get the probability distribution P(term) (owned String version)
    /// Returns an empty vector if total is 0
    #[inline]
    pub fn probability_vector(&self) -> Vec<(String, f64)> {
        if self.total_term_count == 0 {
            return Vec::new();
        }
        let total = self.total_term_count as f64;
        self.term_count
            .iter()
            .map(|(term, &count)| (term.clone(), (count as f64) / total))
            .collect()
    }

    /// Get the probability distribution P(term) (as &str)
    /// Returns an empty vector if total is 0
    #[inline]
    pub fn probability_vector_ref_str(&self) -> Vec<(&str, f64)> {
        if self.total_term_count == 0 {
            return Vec::new();
        }
        let total = self.total_term_count as f64;
        self.term_count
            .iter()
            .map(|(term, &count)| (term.as_str(), (count as f64) / total))
            .collect()
    }

    /// Get the probability P(term) for a specific term
    /// Returns 0.0 if total is 0
    #[inline]
    pub fn probability(&self, term: &str) -> f64 {
        if self.total_term_count == 0 {
            return 0.0;
        }
        (self.term_count(term) as f64) / (self.total_term_count as f64)
    }

    /// Reset all counts
    #[inline]
    pub fn clear(&mut self) {
        self.term_count.clear();
        self.total_term_count = 0;
    }

    /// Shrink internal storage to fit current size
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.term_count.shrink_to_fit();
    }
}