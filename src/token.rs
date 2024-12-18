use std::collections::{hash_map::Keys, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency<IdType> {
    pub token_count: HashMap<String, u64>,
    pub total_token_count: u64,
    pub id: IdType,
}

/// A struct that tracks the frequency of tokens and provides various methods
/// to manipulate and retrieve token frequency data.
///
/// # Type Parameters
/// - `IdType`: The type of the identifier for the token frequency instance.
///
/// # Methods
/// - `new(id: IdType) -> Self`: Creates a new `TokenFrequency` instance with the given identifier.
/// - `new_with_id(id: IdType) -> Self`: Creates a new `TokenFrequency` instance with the given identifier.
/// - `set_id(&mut self, id: IdType) -> &mut Self`: Sets the identifier for the `TokenFrequency` instance.
/// - `add_token(&mut self, token: &str) -> &mut Self`: Adds a token to the frequency count.
/// - `add_token_n(&mut self, token: &str, n: u64) -> &mut Self`: Adds a token to the frequency count with a specified count.
/// - `add_tokens(&mut self, tokens: &[&str]) -> &mut Self`: Adds multiple tokens to the frequency count.
/// - `get_tf_vector(&self) -> Vec<(String, f64)>`: Returns the term frequency (TF) vector.
/// - `get_tf_vector_parallel(&self) -> Vec<(String, f64)>`: Returns the term frequency (TF) vector using parallel processing.
/// - `get_tf_vector_ref(&self) -> Vec<(&str, f64)>`: Returns the term frequency (TF) vector with references.
/// - `get_tf_vector_ref_parallel(&self) -> Vec<(&str, f64)>`: Returns the term frequency (TF) vector with references using parallel processing.
/// - `get_tf_hashmap(&self) -> HashMap<String, f64>`: Returns the term frequency (TF) hashmap.
/// - `get_tf_hashmap_parallel(&self) -> HashMap<String, f64>`: Returns the term frequency (TF) hashmap using parallel processing.
/// - `get_tf_hashmap_ref(&self) -> HashMap<&str, f64>`: Returns the term frequency (TF) hashmap with references.
/// - `get_tf_hashmap_ref_parallel(&self) -> HashMap<&str, f64>`: Returns the term frequency (TF) hashmap with references using parallel processing.
/// - `get_token_tf(&self, token: &str) -> f64`: Returns the term frequency (TF) of a specific token.
/// - `get_token_count_vector(&self) -> Vec<(String, u64)>`: Returns a vector of token counts.
/// - `get_token_count_hashmap(&self) -> HashMap<String, u64>`: Returns a hashmap of token counts.
/// - `get_token_count_hashmap_ref(&self) -> HashMap<&str, u64>`: Returns a hashmap of token counts with references.
/// - `get_total_token_count(&self) -> u64`: Returns the total token count.
/// - `get_total_token_count_ref(&self) -> &u64`: Returns a reference to the total token count.
/// - `get_id(&self) -> &IdType`: Returns a reference to the identifier.
/// - `get_token_count(&self, token: &str) -> u64`: Returns the count of a specific token.
/// - `get_token_count_ref(&self, token: &str) -> &u64`: Returns a reference to the count of a specific token.
/// - `get_most_frequent_token(&self) -> Option<(String, u64)>`: Returns the most frequent token and its count.
/// - `get_most_frequent_token_parallel(&self) -> Option<(String, u64)>`: Returns the most frequent token and its count using parallel processing.
/// - `get_tfidf_vector(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)>`: Returns the TF-IDF vector.
/// - `get_tfidf_vector_parallel(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)>`: Returns the TF-IDF vector using parallel processing.
/// - `contains_token(&self, token: &str) -> bool`: Checks if a token is present in the frequency count.
/// - `get_token_set(&self) -> Vec<String>`: Returns a vector of unique tokens.
/// - `get_token_set_ref(&self) -> Vec<&str>`: Returns a vector of unique tokens with references.
/// - `get_token_hashset(&self) -> HashSet<String>`: Returns a hashset of unique tokens.
/// - `get_token_hashset_ref(&self) -> HashSet<&str>`: Returns a hashset of unique tokens with references.
/// - `get_token_set_len(&self) -> usize`: Returns the number of unique tokens.
/// - `get_token_set_iter(&self) -> Keys<String, u64>`: Returns an iterator over the unique tokens.
/// - `get_token_set_iter_ref(&self) -> impl Iterator<Item = &str>`: Returns an iterator over the unique tokens with references.
/// - `get_token_length_stats(&self) -> Option<(usize, usize, f64)>`: Returns statistics (min, max, average) of token lengths.
/// - `get_token_length_stats_ref(&self) -> Option<(usize, usize, f64)>`: Returns statistics (min, max, average) of token lengths with references.
/// - `get_token_length_stats_parallel(&self) -> Option<(usize, usize, f64)>`: Returns statistics (min, max, average) of token lengths using parallel processing.
/// - `remove_stop_tokens(&mut self, stop_tokens: &[&str])`: Removes stop tokens from the frequency count.
/// - `remove_stop_tokens_parallel(&mut self, stop_tokens: &[&str])`: Removes stop tokens from the frequency count using parallel processing.
/// - `remove_tokens_by_condition<F>(&mut self, condition: F) -> u64 where F: Fn(&str, &u64) -> bool`: Removes tokens based on a condition and returns the total count of removed tokens.
/// - `get_sorted_by_frequency_desc(&self) -> Vec<(String, u64)>`: Returns tokens sorted by frequency in descending order.
/// - `get_sorted_by_frequency_desc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted by frequency in descending order using parallel processing.
/// - `get_sorted_by_frequency_asc(&self) -> Vec<(String, u64)>`: Returns tokens sorted by frequency in ascending order.
/// - `get_sorted_by_frequency_asc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted by frequency in ascending order using parallel processing.
/// - `get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u64)>`: Returns tokens sorted alphabetically in ascending order.
/// - `get_sorted_by_alphabetical_asc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted alphabetically in ascending order using parallel processing.
/// - `get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u64)>`: Returns tokens sorted alphabetically in descending order.
/// - `get_sorted_by_alphabetical_desc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted alphabetically in descending order using parallel processing.
/// - `get_sorted_by_length_desc(&self) -> Vec<(String, u64)>`: Returns tokens sorted by length in descending order.
/// - `get_sorted_by_length_desc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted by length in descending order using parallel processing.
/// - `get_sorted_by_length_asc(&self) -> Vec<(String, u64)>`: Returns tokens sorted by length in ascending order.
/// - `get_sorted_by_length_asc_parallel(&self) -> Vec<(String, u64)>`: Returns tokens sorted by length in ascending order using parallel processing.
/// - `get_unique_token_ratio(&self) -> f64`: Returns the ratio of unique tokens to the total token count.
impl<IdType: Send + Sync> TokenFrequency<IdType> {
    pub fn new(id: IdType) -> Self {
        TokenFrequency {
            id: id,
            token_count: HashMap::new(),
            total_token_count: 0,
        }
    }

    pub fn new_with_id(id: IdType) -> Self {
        TokenFrequency {
            id: id,
            token_count: HashMap::new(),
            total_token_count: 0,
        }
    }

    pub fn set_id(&mut self, id: IdType) -> &mut Self {
        self.id = id;
        self
    }

    pub fn add_token(&mut self, token: &str) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += 1;
        self.total_token_count += 1;
        self
    }

    pub fn add_token_n(&mut self, token: &str, n: u64) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += n;
        self.total_token_count += n;
        self
    }

    pub fn add_tokens(&mut self, tokens: &[&str]) -> &mut Self {
        for &token in tokens {
            let count = self.token_count.entry(token.to_string()).or_insert(0);
            *count += 1;
            self.total_token_count += 1;
        }
        self
    }

    pub fn get_tf_vector(&self) -> Vec<(String, f64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count as f64 / self.total_token_count as f64)
        }).collect()
    }

    pub fn get_tf_vector_parallel(&self) -> Vec<(String, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    pub fn get_tf_vector_ref(&self) -> Vec<(&str, f64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count as f64 / self.total_token_count as f64)
        }).collect()
    }

    pub fn get_tf_vector_ref_parallel(&self) -> Vec<(&str, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.as_str(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    pub fn get_tf_hashmap(&self) -> HashMap<String, f64> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count as f64 / self.total_token_count as f64)
        }).collect()
    }

    pub fn get_tf_hashmap_parallel(&self) -> HashMap<String, f64> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    pub fn get_tf_hashmap_ref(&self) -> HashMap<&str, f64> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count as f64 / self.total_token_count as f64)
        }).collect()
    }

    pub fn get_tf_hashmap_ref_parallel(&self) -> HashMap<&str, f64> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.as_str(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    pub fn get_token_tf(&self, token: &str) -> f64 {
        let count = self.token_count.get(token).copied().unwrap_or(0);
        if self.total_token_count == 0 {
            0.0
        } else {
            count as f64 / self.total_token_count as f64
        }
    }

    pub fn get_token_count_vector(&self) -> Vec<(String, u64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count)
        }).collect()
    }

    pub fn get_token_count_hashmap(&self) -> HashMap<String, u64> {
        self.token_count.clone()
    }

    pub fn get_token_count_hashmap_ref(&self) -> HashMap<&str, u64> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    pub fn get_total_token_count(&self) -> u64 {
        self.total_token_count
    }

    pub fn get_total_token_count_ref(&self) -> &u64 {
        &self.total_token_count
    }

    pub fn get_id(&self) -> &IdType {
        &self.id
    }

    pub fn get_token_count(&self, token: &str) -> u64 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    pub fn get_token_count_ref(&self, token: &str) -> &u64 {
        self.token_count.get(token).unwrap_or(&0)
    }

    pub fn get_most_frequent_token(&self) -> Option<(String, u64)> {
        self.token_count.iter().max_by_key(|&(_, count)| count).map(|(token, &count)| (token.clone(), count))
    }

    pub fn get_most_frequent_token_parallel(&self) -> Option<(String, u64)> {
        self.token_count
            .par_iter()
            .reduce_with(|a, b| if a.1 >= b.1 { a } else { b })
            .map(|(token, &count)| (token.clone(), count))
    }

    pub fn get_tfidf_vector(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = count as f64 / self.total_token_count as f64;
            let idf = idf_map.get(token).copied().unwrap_or(0.0);
            (token.clone(), tf * idf)
        }).collect()
    }

    pub fn get_tfidf_vector_parallel(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = count as f64 / self.total_token_count as f64;
                let idf = idf_map.get(token).copied().unwrap_or(0.0);
                (token.clone(), tf * idf)
            })
            .collect()
    }

    pub fn contains_token(&self, token: &str) -> bool {
        self.token_count.contains_key(token)
    }

    pub fn get_token_set(&self) -> Vec<String> {
        self.token_count.keys().cloned().collect()
    }

    pub fn get_token_set_ref(&self) -> Vec<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_token_hashset(&self) -> HashSet<String> {
        self.token_count.keys().cloned().collect()
    }

    pub fn get_token_hashset_ref(&self) -> HashSet<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_token_set_len(&self) -> usize {
        self.token_count.len()
    }

    pub fn get_token_set_iter(&self) -> Keys<String, u64> {
        self.token_count.keys()
    }

    pub fn get_token_set_iter_ref(&self) -> impl Iterator<Item = &str> {
        self.token_count.keys().map(|s| s.as_str())
    }

    pub fn get_token_length_stats(&self) -> Option<(usize, usize, f64)> {
        if self.token_count.is_empty() {
            return None;
        }

        let lengths: Vec<usize> = self.token_count.keys().map(|token| token.len()).collect();
        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let avg_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;

        Some((min_len, max_len, avg_len))
    }

    pub fn get_token_length_stats_ref(&self) -> Option<(usize, usize, f64)> {
        if self.token_count.is_empty() {
            return None;
        }

        let lengths: Vec<usize> = self.token_count.keys().map(|token| token.len()).collect();
        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let avg_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;

        Some((min_len, max_len, avg_len))
    }

    pub fn get_token_length_stats_parallel(&self) -> Option<(usize, usize, f64)> {
        if self.token_count.is_empty() {
            return None;
        }

        let (min_len, max_len, total_len, count) = self.token_count
            .par_iter()
            .map(|(token, _)| (token.len(), token.len(), token.len(), 1))
            .reduce(
                || (usize::MAX, 0, 0, 0),
                |acc, len| {
                    let min_len = acc.0.min(len.0);
                    let max_len = acc.1.max(len.1);
                    let total_len = acc.2 + len.2;
                    let count = acc.3 + len.3;
                    (min_len, max_len, total_len, count)
                },
            );

        Some((min_len, max_len, total_len as f64 / count as f64))
    }

    pub fn remove_stop_tokens(&mut self, stop_tokens: &[&str]) {
        for &stop_token in stop_tokens {
            if let Some(count) = self.token_count.remove(stop_token) {
                self.total_token_count -= count;
            }
        }
    }

    pub fn remove_stop_tokens_parallel(&mut self, stop_tokens: &[&str]) {
        let to_remove: Vec<String> = stop_tokens
            .par_iter()
            .filter_map(|&stop_token| {
                self.token_count.get(stop_token).map(|_| stop_token.to_string())
            })
            .collect();

        for token in to_remove {
            if let Some(count) = self.token_count.remove(&token) {
                self.total_token_count -= count;
            }
        }
    }

    pub fn remove_tokens_by_condition<F>(&mut self, condition: F) -> u64
    where
        F: Fn(&str, &u64) -> bool,
    {
        let mut removed_total_count = 0;
        self.token_count.retain(|token, count| {
            if condition(token, &count) {
                removed_total_count += *count;
                false
            } else {
                true
            }
        });
        self.total_token_count -= removed_total_count;

        removed_total_count
    }

    pub fn get_sorted_by_frequency_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    pub fn get_sorted_by_frequency_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    pub fn get_sorted_by_frequency_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    pub fn get_sorted_by_frequency_asc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    pub fn get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_asc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    pub fn get_sorted_by_length_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.len().cmp(&b.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_asc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.len().cmp(&b.0.len()));
        token_list
    }

    pub fn get_unique_token_ratio(&self) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        self.token_count.len() as f64 / self.total_token_count as f64
    }
}

