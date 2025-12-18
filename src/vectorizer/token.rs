use core::str;
use std::{collections::{HashMap, HashSet}, fmt::Debug};
use ahash::RandomState;
use serde::{Deserialize, Serialize};

use crate::Corpus;


/// TokenFrequency struct
/// Manages the frequency of token occurrences.
/// Counts the number of times each token appears.
///
/// # Examples
/// ```
/// use crate::tf_idf_vectorizer::vectorizer::token::TokenFrequency;
/// let mut token_freq = TokenFrequency::new();
/// token_freq.add_token("token1");
/// token_freq.add_token("token2");
/// token_freq.add_token("token1");
///
/// assert_eq!(token_freq.token_count("token1"), 2);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency {
    token_count: HashMap<String, u64, RandomState>,
    total_token_count: u64,
}

/// Implementation for adding and removing tokens
impl TokenFrequency {
    /// Create a new TokenFrequency
    pub fn new() -> Self {
        TokenFrequency {
            token_count: HashMap::with_hasher(RandomState::new()),
            total_token_count: 0,
        }
    }

    /// Add a token
    ///
    /// # Arguments
    /// * `token` - Token to add
    #[inline]
    pub fn add_token(&mut self, token: &str) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += 1;
        self.total_token_count += 1;
        self
    }

    /// Add multiple tokens
    ///
    /// # Arguments
    /// * `tokens` - Slice of tokens to add
    #[inline]
    pub fn add_tokens<T>(&mut self, tokens: &[T]) -> &mut Self 
    where T: AsRef<str> 
    {
        for token in tokens {
            let token_str = token.as_ref();
            self.add_token(token_str);
        }
        self
    }

    /// Subtract a token
    ///
    /// # Arguments
    /// * `token` - Token to subtract
    #[inline]
    pub fn sub_token(&mut self, token: &str) -> &mut Self {
        if let Some(count) = self.token_count.get_mut(token) {
            if *count > 1 {
                *count -= 1;
                self.total_token_count -= 1;
            } else if *count == 1 {
                self.token_count.remove(token);
                self.total_token_count -= 1;
            }
        }
        self
    }

    /// Subtract multiple tokens
    ///
    /// # Arguments
    /// * `tokens` - Slice of tokens to subtract
    #[inline]
    pub fn sub_tokens<T>(&mut self, tokens: &[T]) -> &mut Self 
    where T: AsRef<str>
    {
        for token in tokens {
            let token_str = token.as_ref();
            self.sub_token(token_str);
        }
        self
    }

    /// Set the occurrence count for a token
    ///
    /// # Arguments
    /// * `token` - Token
    /// * `count` - Occurrence count
    pub fn set_token_count(&mut self, token: &str, count: u64) -> &mut Self {
        if count == 0 {
            self.token_count.remove(token);
        } else {
            let current_count = self.token_count.entry(token.to_string()).or_insert(0);
            self.total_token_count += count - *current_count;
            *current_count = count;
        }
        self
    }

    /// Merge with another TokenFrequency
    /// # Arguments
    /// * `other` - Another TokenFrequency to merge with
    pub fn add_tokens_from_freq(&mut self, other: &TokenFrequency) -> &mut Self {
        for (token, &count) in &other.token_count {
            let entry = self.token_count.entry(token.clone()).or_insert(0);
            *entry += count;
            self.total_token_count += count;
        }
        self
    }

    /// Scale the token counts by a scalar
    /// # Arguments
    /// * `scalar` - Scalar to scale by
    pub fn scale(&mut self, scalar: f64) -> &mut Self {
        let mut total_count = 0;
        self.token_count.iter_mut().for_each(|(_, count)| {
            *count = ((*count as f64) * scalar).round() as u64;
            total_count += *count;
        });
        self.total_token_count = total_count;
        self
    }
}

impl<T> From<&[T]> for TokenFrequency
where
    T: AsRef<str>,
{
    fn from(tokens: &[T]) -> Self {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(tokens);
        tf
    }
}

impl From<Corpus> for TokenFrequency {
    fn from(corpus: Corpus) -> Self {
        let mut tf = TokenFrequency::new();
        for entry in corpus.token_counts.iter() {
            let token = entry.key();
            let count = *entry.value();
            tf.set_token_count(token, count);
        }
        tf
    }
}

/// Implementation for retrieving information from TokenFrequency
impl TokenFrequency {
    /// Get a vector of all tokens and their counts
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of tokens and their counts
    #[inline]
    pub fn token_count_vector(&self) -> Vec<(String, u64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count)
        }).collect()
    }

    /// Get a vector of all tokens and their counts (as &str)
    ///
    /// # Returns
    /// * `Vec<(&str, u64)>` - Vector of tokens and their counts
    #[inline]
    pub fn token_count_vector_ref_str(&self) -> Vec<(&str, u64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    /// Get a hashmap of all tokens and their counts (as &str)
    ///
    /// # Returns
    /// * `HashMap<&str, u64>` - HashMap of tokens and their counts
    #[inline]
    pub fn token_count_hashmap_ref_str(&self) -> HashMap<&str, u64, RandomState> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    /// Get the total count of all tokens
    ///
    /// # Returns
    /// * `u64` - Total token count
    #[inline]
    pub fn token_sum(&self) -> u64 {
        self.total_token_count
    }

    /// Get the occurrence count for a specific token
    ///
    /// # Arguments
    /// * `token` - Token
    ///
    /// # Returns
    /// * `u64` - Occurrence count for the token
    #[inline]
    pub fn token_count(&self, token: &str) -> u64 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    /// Get the most frequent tokens
    /// If multiple tokens have the same count, all are returned
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of most frequent tokens and their counts
    #[inline]
    pub fn most_frequent_tokens_vector(&self) -> Vec<(String, u64)> {
        if let Some(&max_count) = self.token_count.values().max() {
            self.token_count.iter()
                .filter(|&(_, &count)| count == max_count)
                .map(|(token, &count)| (token.clone(), count))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the count of the most frequent token
    ///
    /// # Returns
    /// * `u64` - Count of the most frequent token
    #[inline]
    pub fn most_frequent_token_count(&self) -> u64 {
        if let Some(&max_count) = self.token_count.values().max() {
            max_count
        } else {
            0
        }
    }

    /// Check if a token exists
    ///
    /// # Arguments
    /// * `token` - Token
    ///
    /// # Returns
    /// * `bool` - true if the token exists, false otherwise
    #[inline]
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_count.contains_key(token)
    }

    /// Get the set of tokens
    ///
    /// # Returns
    /// * `Vec<String>` - Set of tokens
    #[inline]
    pub fn token_set(&self) -> Vec<String> {
        self.token_count.keys().cloned().collect()
    }

    /// Get the set of tokens (as &str)
    ///
    /// # Returns
    /// * `Vec<&str>` - Set of tokens
    #[inline]
    pub fn token_set_ref_str(&self) -> Vec<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    /// Get the set of tokens as a HashSet
    ///
    /// # Returns
    /// * `HashSet<String>` - Set of tokens
    #[inline]
    pub fn token_hashset(&self) -> HashSet<String, RandomState> {
        self.token_count.keys().cloned().collect()
    }

    /// Get the set of tokens as a HashSet (as &str)
    ///
    /// # Returns
    /// * `HashSet<&str>` - Set of tokens
    #[inline]
    pub fn token_hashset_ref_str(&self) -> HashSet<&str, RandomState> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of unique tokens
    ///
    /// # Returns
    /// * `usize` - Number of unique tokens
    #[inline]
    pub fn token_num(&self) -> usize {
        self.token_count.len()
    }

    /// Remove stop tokens
    ///
    /// # Arguments
    /// * `stop_tokens` - Slice of stop tokens to remove
    ///
    /// # Returns
    /// * `u64` - Total count of removed tokens
    #[inline]
    pub fn remove_stop_tokens(&mut self, stop_tokens: &[&str]) -> u64{
        let mut removed_total_count: u64 = 0;
        for &stop_token in stop_tokens {
            if let Some(count) = self.token_count.remove(stop_token) {
                removed_total_count += count as u64;
            }
        }
        self.total_token_count -= removed_total_count;
        removed_total_count
    }

    /// Remove tokens by a condition
    ///
    /// # Arguments
    /// * `condition` - Closure to determine which tokens to remove
    ///
    /// # Returns
    /// * `u64` - Total count of removed tokens
    #[inline]
    pub fn remove_tokens_by<F>(&mut self, condition: F) -> u64
    where
        F: Fn(&str, &u64) -> bool,
    {
        let mut removed_total_count: u64 = 0;
        self.token_count.retain(|token, count| {
            if condition(token, count) {
                removed_total_count += *count as u64;
                false
            } else {
                true
            }
        });
        self.total_token_count -= removed_total_count as u64;

        removed_total_count
    }

    /// Get a vector of tokens sorted by frequency (descending)
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of tokens sorted by frequency
    #[inline]
    pub fn sorted_frequency_vector(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    /// Get a vector of tokens sorted by dictionary order (ascending)
    ///
    /// # Returns
    /// * `Vec<(String, u64)>` - Vector of tokens sorted by dictionary order
    #[inline]
    pub fn sorted_dict_order_vector(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    /// Calculate the diversity of tokens
    /// 1.0 indicates complete diversity, 0.0 indicates no diversity
    ///
    /// # Returns
    /// * `f64` - Diversity of tokens
    #[inline]
    pub fn unique_token_ratio(&self) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        self.token_count.len() as f64 / self.total_token_count as f64
    }

    /// Get the probability distribution P(token) (owned String version)
    /// Returns an empty vector if total is 0
    #[inline]
    pub fn probability_vector(&self) -> Vec<(String, f64)> {
        if self.total_token_count == 0 {
            return Vec::new();
        }
        let total = self.total_token_count as f64;
        self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), (count as f64) / total))
            .collect()
    }

    /// Get the probability distribution P(token) (as &str)
    /// Returns an empty vector if total is 0
    #[inline]
    pub fn probability_vector_ref_str(&self) -> Vec<(&str, f64)> {
        if self.total_token_count == 0 {
            return Vec::new();
        }
        let total = self.total_token_count as f64;
        self.token_count
            .iter()
            .map(|(token, &count)| (token.as_str(), (count as f64) / total))
            .collect()
    }

    /// Get the probability P(token) for a specific token
    /// Returns 0.0 if total is 0
    #[inline]
    pub fn probability(&self, token: &str) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        (self.token_count(token) as f64) / (self.total_token_count as f64)
    }

    /// Reset all counts
    #[inline]
    pub fn clear(&mut self) {
        self.token_count.clear();
        self.total_token_count = 0;
    }

    /// Shrink internal storage to fit current size
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.token_count.shrink_to_fit();
    }
}