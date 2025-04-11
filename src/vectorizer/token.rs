use core::str;
use std::{collections::{hash_map::Keys, HashMap, HashSet}, fmt::Debug};

use fst::Map;
use num::Num;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::utils::scaler::{AttachedNormalizer, Normalizer};

/*
/// paralel操作は順序が保証されないため、順序が重要な場合は注意が必要
*/

///  TokenFrequency 構造体
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency {
    pub token_count: HashMap<String, u32>,
    pub total_token_count: u64,
}

/// Tokenの追加、削除の実装
impl TokenFrequency {
    pub fn new() -> Self {
        TokenFrequency {
            token_count: HashMap::new(),
            total_token_count: 0,
        }
    }

    /// tokenを追加する
    /// 
    /// # Arguments
    /// * `token` - 追加するトークン
    #[inline(always)]
    pub fn add_token(&mut self, token: &str) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += 1;
        self.total_token_count += 1;
        self
    }

    /// 複数のtokenを追加する
    /// 
    /// # Arguments
    /// * `tokens` - 追加するトークンのスライス
    #[inline(always)]
    pub fn add_tokens<T>(&mut self, tokens: &[T]) -> &mut Self 
    where T: AsRef<str> 
    {
        for token in tokens {
            let token_str = token.as_ref();
            self.add_token(token_str);
        }
        self
    }

    /// tokenを引く
    /// 
    /// # Arguments
    /// * `token` - 引くトークン
    #[inline(always)]
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

    /// 複数のtokenを引く
    /// 
    /// # Arguments
    /// * `tokens` - 引くトークンのスライス
    #[inline(always)]
    pub fn sub_tokens<T>(&mut self, tokens: &[T]) -> &mut Self 
    where T: AsRef<str>
    {
        for token in tokens {
            let token_str = token.as_ref();
            self.sub_token(token_str);
        }
        self
    }

    /// tokenの出現回数を指定する
    /// 
    /// # Arguments
    /// * `token` - トークン
    /// * `count` - 出現回数
    pub fn set_token_count(&mut self, token: &str, count: u32) -> &mut Self {
        if let Some(existing_count) = self.token_count.get_mut(token) {
            if count >= *existing_count {
                self.total_token_count += count as u64 - *existing_count as u64;
            } else {
                self.total_token_count -= *existing_count as u64 - count as u64;
            }
            *existing_count = count;
        } else {
            self.token_count.insert(token.to_string(), count);
            self.total_token_count += count as u64;
        }
        self
    }
}

/// TF-calculationの実装
impl TokenFrequency
{
    #[inline(always)]
    pub fn tf_calc(max_count: u32, count: u32) -> f64 {
        (count as f64 + 1.0).ln() / (max_count as f64 + 1.0).ln()
    }

    // Vec<(String, u16)>を取得
    #[inline(always)]
    pub fn tf_vector<N>(&self) -> Vec<(String, N)> 
    where f64: Normalizer<N>, N: Num {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc(max_count, count).into_normalized())
            })
            .collect()
    }

    // Vec<(&str, u16)>を取得
    #[inline(always)]
    pub fn tf_vector_ref_str<N>(&self) -> Vec<(&str, N)>
    where f64: Normalizer<N>, N: Num {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc(max_count, count).into_normalized())
            })
            .collect()
    }

    // HashMap<String, u16>を取得
    #[inline(always)]
    pub fn tf_hashmap<N>(&self) -> HashMap<String, N> 
    where f64: Normalizer<N>, N: Num {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc(max_count, count).into_normalized())
            })
            .collect()
    }

    // HashMap<&str, u16>を取得
    #[inline(always)]
    pub fn tf_hashmap_ref_str<N>(&self) -> HashMap<&str, N> 
    where f64: Normalizer<N>, N: Num {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc(max_count, count).into_normalized())
            })
            .collect()
    }

    // 特定のトークンのTFを取得
    #[inline(always)]
    pub fn tf_token<N>(&self, token: &str) -> N 
    where f64: Normalizer<N>, N: Num{
        let max_count = self.get_most_frequent_token_count();
        let count = self.token_count.get(token).copied().unwrap_or(0);
        Self::tf_calc(max_count, count).into_normalized()
    }
}

/// IDF-calculationの実装
impl TokenFrequency {
    #[inline(always)]
    pub fn idf_max(&self, total_doc_count: u64) -> f64 {
        (1.0 + total_doc_count as f64 / (2.0)).ln()
    }

    #[inline(always)]
    pub fn idf_calc(total_doc_count: u64, max_idf: f64, doc_count: u32) -> f64 {
        (1.0 + total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf
    }

    #[inline(always)]
    pub fn idf_vector<N>(&self, total_doc_count: u64) -> Vec<(String, N)> 
    where f64: Normalizer<N>, N: Num {
        self.token_count
            .iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf.into_normalized())
            })
            .collect()
    }

    #[inline(always)]
    pub fn idf_vector_ref_str<N>(&self, total_doc_count: u64) -> Vec<(&str, N)> 
    where f64: Normalizer<N>, N: Num {
        self.token_count.iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf.into_normalized())
        }).collect()
    }

    #[inline(always)]
    pub fn idf_hashmap<N>(&self, total_doc_count: u64) -> HashMap<String, N> 
    where f64: Normalizer<N>, N: Num {
        self.token_count
            .iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf.into_normalized())
            })
            .collect()
    }

    #[inline(always)]
    pub fn idf_hashmap_ref_str<N>(&self, total_doc_count: u64) -> HashMap<&str, N> 
    where f64: Normalizer<N>, N: Num {
        self.token_count.iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf.into_normalized())
        }).collect()
    }
}

/// TokenFrequencyの情報を取得するための実装
impl TokenFrequency {
    #[inline(always)]
    pub fn token_count_vector(&self) -> Vec<(String, u32)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count)
        }).collect()
    }

    #[inline(always)]
    pub fn token_count_vector_ref_str(&self) -> Vec<(&str, u32)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    #[inline(always)]
    pub fn token_count_hashmap(&self) -> HashMap<String, u32> {
        self.token_count.clone()
    }

    #[inline(always)]
    pub fn token_count_hashmap_ref_str(&self) -> HashMap<&str, u32> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    #[inline(always)]
    pub fn token_total_count(&self) -> u64 {
        self.total_token_count
    }

    #[inline(always)]
    pub fn token_count(&self, token: &str) -> u32 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    /// 最も頻繁に出現するトークンたち
    #[inline(always)]
    pub fn most_frequent_tokens_vector(&self) -> Vec<(String, u32)> {
        if let Some(&max_count) = self.token_count.values().max() {
            self.token_count.iter()
                .filter(|&(_, &count)| count == max_count)
                .map(|(token, &count)| (token.clone(), count))
                .collect()
        } else {
            Vec::new()
        }
    }

    #[inline(always)]
    pub fn most_frequent_token_count(&self) -> u32 {
        if let Some(&max_count) = self.token_count.values().max() {
            max_count
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_count.contains_key(token)
    }

    #[inline(always)]
    pub fn token_set(&self) -> Vec<String> {
        self.token_count.keys().cloned().collect()
    }

    #[inline(always)]
    pub fn token_set_ref_str(&self) -> Vec<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    #[inline(always)]
    pub fn token_hashset(&self) -> HashSet<String> {
        self.token_count.keys().cloned().collect()
    }

    #[inline(always)]
    pub fn token_hashset_ref_str(&self) -> HashSet<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    #[inline(always)]
    pub fn token_num(&self) -> usize {
        self.token_count.len()
    }

    #[inline(always)]
    pub fn remove_stop_tokens(&mut self, stop_tokens: &[&str]) {
        for &stop_token in stop_tokens {
            if let Some(count) = self.token_count.remove(stop_token) {
                self.total_token_count -= count as u64;
            }
        }
    }

    #[inline(always)]
    pub fn remove_tokens_by_condition<F>(&mut self, condition: F) -> u64
    where
        F: Fn(&str, &u32) -> bool,
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

    #[inline(always)]
    pub fn sorted_frequency_vector(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_frequency_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_frequency_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_frequency_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_alphabetical_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_alphabetical_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_length_desc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_length_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_length_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.len().cmp(&b.0.len()));
        token_list
    }

    #[inline(always)]
    pub fn get_sorted_by_length_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.len().cmp(&b.0.len()));
        token_list
    }

    #[inline(always)]
    pub fn get_unique_token_ratio(&self) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        self.token_count.len() as f64 / self.total_token_count as f64
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.token_count.clear();
        self.total_token_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_token() {
        let mut tf = TokenFrequency::new();
        tf.add_token("rust");
        assert_eq!(tf.token_count.get("rust"), Some(&1));
        assert_eq!(tf.total_token_count, 1);
    }

    #[test]
    fn test_add_tokens() {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&["rust", "rust", "programming"]);
        assert_eq!(tf.token_count.get("rust"), Some(&2));
        assert_eq!(tf.token_count.get("programming"), Some(&1));
        assert_eq!(tf.total_token_count, 3);
    }

    #[test]
    fn test_sub_token() {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&["rust", "rust", "programming"]);
        tf.sub_token("rust");
        assert_eq!(tf.token_count.get("rust"), Some(&1));
        assert_eq!(tf.total_token_count, 2);
    }

    #[test]
    fn test_tfidf_calc() {
        let tfidf = TokenFrequency::tfidf_calc(2.0, 1.5);
        assert_eq!(tfidf, 3.0);
    }

    #[test]
    fn test_reset() {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&["rust", "programming"]);
        tf.reset();
        assert!(tf.token_count.is_empty());
        assert_eq!(tf.total_token_count, 0);
    }

    #[test]
    fn test_get_token_length_stats() {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&["rust", "go", "java"]);
        let stats = tf.get_token_length_stats();
        assert_eq!(stats, Some((2, 4, 3.3333333333333335)));
    }

    #[test]
    fn test_unique_token_ratio() {
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&["rust", "rust", "go"]);
        assert_eq!(tf.get_unique_token_ratio(), 2.0 / 3.0);
    }
}