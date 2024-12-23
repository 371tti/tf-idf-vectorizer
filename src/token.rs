use core::str;
use std::{collections::{hash_map::Keys, HashMap, HashSet}, f32::consts::E, fmt::Debug, hash::Hash, sync::{atomic::{AtomicU64, Ordering}, Arc, Mutex}};

use fst::{Map, MapBuilder, Streamer};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use sprs::CsVec;

/*
/// paralel操作は順序が保証されないため、順序が重要な場合は注意が必要
*/


///  FromVec トレイト
pub trait FromVec<T> {
    fn from_vec(vec: Vec<T>) -> Self;
    fn from_vec_parallel(vec: Vec<T>) -> Self;
}

//  CsVec<T> に対する FromVec トレイトの実装
impl<T> FromVec<T> for CsVec<T>
where
    T: Clone + PartialEq + Default + Send + Sync,
{
    /// 通常のfrom_vec実装
    fn from_vec(vec: Vec<T>) -> Self {
        let indices: Vec<usize> = vec
            .iter()
            .enumerate()
            .filter(|&(_, value)| value != &T::default())
            .map(|(index, _)| index)
            .collect();

        let values: Vec<T> = vec
            .iter()
            .filter(|value| **value != T::default())
            .cloned()
            .collect();

        CsVec::new(vec.len(), indices, values)
    }

    /// 並列処理版のfrom_vec実装 順序が保証されない
    fn from_vec_parallel(vec: Vec<T>) -> Self {
        // インデックスと値を並列に収集
        let pairs: Vec<(usize, T)> = vec
            .into_par_iter() // 並列イテレータに変換
            .enumerate()
            .filter(|&(_, ref value)| *value != T::default())
            .collect();

        // インデックスと値を分離
        let (indices, values): (Vec<usize>, Vec<T>) = pairs.into_iter().unzip();

        CsVec::new(indices.len(), indices, values)
    }
}


///  TokenFrequency 構造体
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency {
    pub token_count: HashMap<String, u32>,
    pub total_token_count: u64,
}


impl TokenFrequency {

    pub fn new() -> Self {
        TokenFrequency {
            token_count: HashMap::new(),
            total_token_count: 0,
        }
    }

    pub fn add_token(&mut self, token: &str) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += 1;
        self.total_token_count += 1;
        self
    }

    pub fn add_token_n(&mut self, token: &str, n: u32) -> &mut Self {
        let count = self.token_count.entry(token.to_string()).or_insert(0);
        *count += n;
        self.total_token_count += n as u64;
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

    pub fn add_tokens_string(&mut self, tokens: &[String]) -> &mut Self {
        for token in tokens {
            let count = self.token_count.entry(token.clone()).or_insert(0);
            *count += 1;
            self.total_token_count += 1;
        }
        self
    }

    pub fn sub_token(&mut self, token: &str) -> &mut Self {
        if let Some(count) = self.token_count.get_mut(token) {
            if *count > 0 {
                *count -= 1;
                self.total_token_count -= 1;
            }
        }
        self
    }

    pub fn sub_token_n(&mut self, token: &str, n: u32) -> &mut Self {
        if let Some(count) = self.token_count.get_mut(token) {
            if *count >= n {
                *count -= n;
                self.total_token_count -= n as u64;
            }
        }
        self
    }

    pub fn sub_tokens(&mut self, tokens: &[&str]) -> &mut Self {
        for &token in tokens {
            if let Some(count) = self.token_count.get_mut(token) {
                if *count > 0 {
                    *count -= 1;
                    self.total_token_count -= 1;
                }
            }
        }
        self
    }

    pub fn sub_tokens_string(&mut self, tokens: &[String]) -> &mut Self {
        for token in tokens {
            if let Some(count) = self.token_count.get_mut(token.as_str()) {
                if *count > 0 {
                    *count -= 1;
                    self.total_token_count -= 1;
                }
            }
        }
        self
    }

    #[inline]
    pub fn tf_calc(max_count: u32, count: u32) -> f64 {
        (count as f64 + 1.0).ln() / (max_count as f64 + 1.0).ln()
    }

    #[inline]
    pub fn tf_calc_as_u16(max_count: u32, count: u32) -> u16 {
        let normalized_value = (count as f64 + 1.0).ln() / (max_count as f64 + 1.0).ln();
        // 0～65535 にスケール
        (normalized_value * 65535.0).round() as u16
    }
    
    #[inline]
    pub fn tf_calc_as_u32(max_count: u32, count: u32) -> u32 {
        let normalized_value = (count as f64 + 1.0).ln() / (max_count as f64 + 1.0).ln();
        // 0～4294967295 にスケール
        (normalized_value * 4294967295.0).round() as u32
    }

    // Vec<(String, u16)>を取得
    pub fn get_tf_vector(&self) -> Vec<(String, u16)> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // 並列処理でVec<(String, u16)>を取得
    pub fn get_tf_vector_parallel(&self) -> Vec<(String, u16)> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // Vec<(&str, u16)>を取得
    pub fn get_tf_vector_ref(&self) -> Vec<(&str, u16)> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // 並列処理でVec<(&str, u16)>を取得
    pub fn get_tf_vector_ref_parallel(&self) -> Vec<(&str, u16)> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // HashMap<String, u16>を取得
    pub fn get_tf_hashmap(&self) -> HashMap<String, u16> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // 並列処理でHashMap<String, u16>を取得
    pub fn get_tf_hashmap_parallel(&self) -> HashMap<String, u16> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                (token.clone(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // HashMap<&str, u16>を取得
    pub fn get_tf_hashmap_ref(&self) -> HashMap<&str, u16> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // 並列処理でHashMap<&str, u16>を取得
    pub fn get_tf_hashmap_ref_parallel(&self) -> HashMap<&str, u16> {
        let max_count = self.get_most_frequent_token_count();
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                (token.as_str(), Self::tf_calc_as_u16(max_count, count))
            })
            .collect()
    }

    // 特定のトークンのTFを取得
    pub fn get_token_tf(&self, token: &str) -> u16 {
        let max_count = self.get_most_frequent_token_count();
        let count = self.token_count.get(token).copied().unwrap_or(0);
        Self::tf_calc_as_u16(max_count, count)
    }

    #[inline]
    pub fn idf_max(&self, total_doc_count: u64) -> f64 {
        (1.0 + total_doc_count as f64 / (2.0)).ln()
    }

    #[inline]
    pub fn idf_calc(total_doc_count: u64, max_idf: f64, doc_count: u32) -> f64 {
        (1.0 + total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf
    }

    #[inline]
    pub fn idf_calc_as_u16(total_doc_count: u64, max_idf: f64, doc_count: u32) -> u16 {
        let normalized_value = (1.0 + total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf;
        // 0～65535 にスケール
        (normalized_value * 65535.0).round() as u16
    }

    #[inline]
    pub fn idf_calc_as_u32(total_doc_count: u64, max_idf: f64, doc_count: u32) -> u32 {
        let normalized_value = (1.0 + total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf;
        // 0～4294967295 にスケール
        (normalized_value * 4294967295.0).round() as u32
    }

    pub fn get_idf_vector(&self, total_doc_count: u64) -> Vec<(String, u16)> {
        self.token_count
            .iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf)
            })
            .collect()
    }

    pub fn get_idf_vector_ref(&self, total_doc_count: u64) -> Vec<(&str, u16)> {
        self.token_count.iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf)
        }).collect()
    }

    pub fn get_idf_vector_parallel(&self, total_doc_count: u64) -> Vec<(String, u16)> {
        self.token_count
            .par_iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf)
            })
            .collect()
    }

    pub fn get_idf_vector_ref_parallel(&self, total_doc_count: u64) -> Vec<(&str, u16)> {
        self.token_count.par_iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf)
        }).collect()
    }

    pub fn get_idf_hashmap(&self, total_doc_count: u64) -> HashMap<String, u16> {
        self.token_count
            .iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf)
            })
            .collect()
    }

    pub fn get_idf_hashmap_ref(&self, total_doc_count: u64) -> HashMap<&str, u16> {
        self.token_count.iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf)
        }).collect()
    }

    pub fn get_idf_hashmap_parallel(&self, total_doc_count: u64) -> HashMap<String, u16> {
        self.token_count
            .par_iter()
            .map(|(token, &doc_count)| {
                let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
                (token.clone(), idf)
            })
            .collect()
    }

    pub fn get_idf_hashmap_ref_parallel(&self, total_doc_count: u64) -> HashMap<&str, u16> {
        self.token_count.par_iter().map(|(token, &doc_count)| {
            let idf = Self::idf_calc_as_u16(total_doc_count, self.idf_max(total_doc_count), doc_count);
            (token.as_str(), idf)
        }).collect()
    }

    pub fn get_token_count_vector(&self) -> Vec<(String, u32)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count)
        }).collect()
    }

    pub fn get_token_count_hashmap(&self) -> HashMap<String, u32> {
        self.token_count.clone()
    }

    pub fn get_token_count_hashmap_ref(&self) -> HashMap<&str, u32> {
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

    pub fn get_token_count(&self, token: &str) -> u32 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    pub fn get_token_count_ref(&self, token: &str) -> &u32 {
        self.token_count.get(token).unwrap_or(&0)
    }

    pub fn get_most_frequent_tokens(&self) -> Vec<(String, u32)> {
        if let Some(&max_count) = self.token_count.values().max() {
            self.token_count.iter()
                .filter(|&(_, &count)| count == max_count)
                .map(|(token, &count)| (token.clone(), count))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_most_frequent_token_count(&self) -> u32 {
        if let Some(&max_count) = self.token_count.values().max() {
            max_count
        } else {
            0
        }
    }

    pub fn get_most_frequent_tokens_parallel(&self) -> Vec<(String, u32)> {
        if self.token_count.is_empty() {
            return Vec::new();
        }
        let max_frequency = self
            .token_count
            .par_iter()
            .map(|(_, &count)| count)
            .max()
            .unwrap();
        self.token_count
            .par_iter()
            .filter(|&(_, &count)| count == max_frequency)
            .map(|(token, &count)| (token.clone(), count))
            .collect()
    }

    pub fn tfidf_calc(tf : f64, idf: f64) -> f64 {
        tf * idf
    }

    pub fn tfidf_calc_as_u16(tf : u16, idf: u16) -> u16 {
        let normalized_value = (tf as f64 * idf as f64) / u32::MAX as f64;
        // 0～65535 にスケール
        (normalized_value * 65535.0).ceil() as u16
    }

    pub fn tfidf_calc_as_u32(tf : u32, idf: u32) -> u32 {
        let normalized_value = (tf as f64 * idf as f64) / u32::MAX as f64;
        // 0～4294967295 にスケール
        (normalized_value * 4294967295.0).ceil() as u32
    }

    pub fn get_tfidf_vector(&self, idf_map: &HashMap<String, u16>) -> Vec<(String, u16)> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
            let idf = idf_map.get(token).copied().unwrap_or(0);
            (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
        }).collect()
    }

    pub fn get_tfidf_vector_fst(&self, idf_map: &Map<Vec<u8>>) -> Vec<(String, u16)> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
            let idf = match idf_map.get(token.as_bytes()) {
                Some(idf) => idf as u16,
                None => 0,
            };
            (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
        }).collect()
    }

    pub fn get_tfidf_hashmap(&self, idf_map: &HashMap<String, u16>) -> HashMap<String, u16> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
            let idf = idf_map.get(token).copied().unwrap_or(0);
            (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
        }).collect()
    }

    pub fn get_tfidf_hashmap_fst(&self, idf_map: &Map<Vec<u8>>) -> HashMap<String, u16> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
            let idf = match idf_map.get(token.as_bytes()) {
                Some(idf) => idf as u16,
                None => 0,
            };
            (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
        }).collect()
    }

    pub fn get_tfidf_vector_parallel(&self, idf_map: &HashMap<String, u16>) -> Vec<(String, u16)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
                let idf = idf_map.get(token).copied().unwrap_or(0);
                (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
            })
            .collect()
    }

    pub fn get_tfidf_vector_fst_parallel(&self, idf_map: &Map<Vec<u8>>) -> Vec<(String, u16)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
                let idf = match idf_map.get(token.as_bytes()) {
                    Some(idf) => idf as u16,
                    None => 0,
                };
                (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
            })
            .collect()
    }

    pub fn get_tfidf_hashmap_parallel(&self, idf_map: &HashMap<String, u16>) -> HashMap<String, u16> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
                let idf = idf_map.get(token).copied().unwrap_or(0);
                (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
            })
            .collect()
    }

    pub fn get_tfidf_hashmap_fst_parallel(&self, idf_map: &Map<Vec<u8>>) -> HashMap<String, u16> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
                let idf = match idf_map.get(token.as_bytes()) {
                    Some(idf) => idf as u16,
                    None => 0,
                };
                (token.clone(), Self::tfidf_calc_as_u16(tf, idf))
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

    pub fn get_token_set_iter(&self) -> Keys<String, u32> {
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
                self.total_token_count -= count as u64;
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
                self.total_token_count -= count as u64;
            }
        }
    }

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

    pub fn get_sorted_by_frequency_desc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    pub fn get_sorted_by_frequency_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    pub fn get_sorted_by_frequency_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    pub fn get_sorted_by_frequency_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    pub fn get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    pub fn get_sorted_by_alphabetical_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    pub fn get_sorted_by_length_desc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_desc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_asc(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.len().cmp(&b.0.len()));
        token_list
    }

    pub fn get_sorted_by_length_asc_parallel(&self) -> Vec<(String, u32)> {
        let mut token_list: Vec<(String, u32)> = self.token_count
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

    pub fn reset(&mut self) {
        self.token_count.clear();
        self.total_token_count = 0;
    }
}



pub struct Index<IdType> 
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    // doc_id -> (圧縮ベクトル, 文書の総トークン数)
    // 圧縮ベクトル: インデックス順にトークンの TF を保持
    pub index: HashMap<IdType, (CsVec<u16>, u64/*token num */)>,
    pub avg_tokens_len: u64,     // 全文書の平均トークン長
    pub max_tokens_len: u64,     // 全文書の最大トークン長
    pub idf: Map<Vec<u8>>,       // fst::Map 形式の IDF
    pub total_doc_count: u64,    // 文書総数
}

impl<IdType> Index<IdType>
where
    IdType: Clone + Eq + std::hash::Hash + Serialize,
{
    
    pub fn new_with_set(index: HashMap<IdType, (CsVec<u16>, u64/*token num */)>, idf: Map<Vec<u8>>, avg_tokens_len: u64, max_tokens_len: u64, total_doc_count: u64) -> Self {
        Self {
            index,
            idf,
            avg_tokens_len,
            max_tokens_len,
            total_doc_count,
        }
    }

    pub fn get_index(&self) -> &HashMap<IdType, (CsVec<u16>, u64/*token num */)> {
        &self.index
    }

    pub fn search_cosin_similarity(&self, query: &[&str], n: usize) -> Vec<(&IdType, f64)> {
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cosine similarityで検索
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, document)| {
            let similarity = Self::cosine_similarity(&document.0, &query_csvec);
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    pub fn search_cosin_similarity_tuned(&self, query: &[&str], n: usize, b:f64) -> Vec<(&IdType, f64)> {
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cosine similarityで検索
        let max_for_len_norm = (self.max_tokens_len as f64 / self.avg_tokens_len as f64);
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, (document, doc_len))| {
            let len_norm = 0.5 + ((((*doc_len as f64 / self.avg_tokens_len as f64) / max_for_len_norm) - 0.5) * b);
            let similarity = Self::cosine_similarity(document, &query_csvec) * len_norm;
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    fn cosine_similarity(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> f64 {
        // 内積を計算
        let dot_product = Self::dot_product_u16(vec_a, vec_b) as f64;
        
        // ノルム（ベクトルの長さ）を計算
        let norm_a = (Self::dot_product_u16(vec_a, vec_a) as f64).sqrt();
        let norm_b = (Self::dot_product_u16(vec_b, vec_b) as f64).sqrt();
    
        // コサイン類似度を返す
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0 // 少なくとも一方のベクトルがゼロの場合
        }
    }

    fn dot_product_u16(vec_a: &CsVec<u16>, vec_b: &CsVec<u16>) -> u64 {
        let mut result: u64 = 0;
    
        // `CsVec` のインデックスと値をイテレート
        let mut iter_a = vec_a.iter();
        let mut iter_b = vec_b.iter();
    
        let mut a = iter_a.next();
        let mut b = iter_b.next();
    
        while let (Some((index_a, &val_a)), Some((index_b, &val_b))) = (a, b) {
            match index_a.cmp(&index_b) {
                std::cmp::Ordering::Equal => {
                    // 両方のインデックスが一致
                    result = result.saturating_add((val_a as u64) * (val_b as u64));
                    a = iter_a.next();
                    b = iter_b.next();
                }
                std::cmp::Ordering::Less => {
                    // vec_a の次へ進む
                    a = iter_a.next();
                }
                std::cmp::Ordering::Greater => {
                    // vec_b の次へ進む
                    b = iter_b.next();
                }
            }
        }
    
        result
    }
 
    pub fn search_bm25_tfidf(&self, query: &[&str], n: usize, k1: f64, b: f64) -> Vec<(&IdType, f64)> {
        println!("{:?}", query);
        //  queryのtfを生成
        let mut binding = TokenFrequency::new();
        let query_tf = binding.add_tokens(query);

        //  idfからqueryのtfidfを生成
        let query_tfidf_vec: HashMap<String, u16> = query_tf.get_tfidf_hashmap_fst_parallel(&self.idf);
        let mut sorted_query_tfidf_vec: Vec<u16> = Vec::new();
        let mut stream = self.idf.stream();
        while let Some((token, idf)) = stream.next() {
            let tfidf = *query_tfidf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
            sorted_query_tfidf_vec.push(tfidf);
        }
        let query_csvec: CsVec<u16> = CsVec::from_vec(sorted_query_tfidf_vec);

        //  cosine similarityで検索
        let mut similarities: Vec<(&IdType, f64)> = self
        .index
        .iter()
        .filter_map(|(id, document)| {
            let similarity = Self::bm25_with_csvec(
                &query_csvec,
                &document.0,
                 document.1, 
                 self.avg_tokens_len as f64, 
                 k1, 
                 b);
            if similarity > 0.0 {
                Some((id, similarity))
            } else {
                None
            }
        })
        .collect();

        // 類似度で降順ソートし、上位 n 件を取得
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        similarities

    }

    fn bm25_with_csvec(
        query_vec: &CsVec<u16>, // クエリのTF-IDFベクトル（u16）
        doc_vec: &CsVec<u16>,   // 文書のTF-IDFベクトル（u16）
        doc_len: u64,           // 文書のトークン数
        avg_doc_len: f64,       // 平均文書長
        k1: f64,                // BM25のパラメータ
        b: f64,                 // 文書長補正のパラメータ
    ) -> f64 {
        let mut score = 0.0;
    
        // 文書長補正を計算
        let len_norm: f64 = 1.0 - b + b * (doc_len as f64 / avg_doc_len);
    
        // `u16` の最大値
        let max_u16: f64 = u16::MAX as f64;
    
        // ベクトルのインデックスと値を効率的に走査
        let mut query_iter = query_vec.iter();
        let mut doc_iter = doc_vec.iter();
    
        let mut query = query_iter.next();
        let mut doc = doc_iter.next();
    
        while let (Some((query_index, &query_value)), Some((doc_index, &doc_value))) = (query, doc) {
            match query_index.cmp(&doc_index) {
                std::cmp::Ordering::Equal => {
                    // 両方に存在するトークンの場合
                    let tf_f = doc_value as f64 / max_u16; // 文書内TF-IDF
                    let idf_f = query_value as f64 / max_u16; // クエリのTF-IDF（IDF含む）
    
                    let numerator = tf_f * (k1 + 1.0);
                    let denominator = tf_f + k1 * len_norm;
    
                    score += idf_f * (numerator / denominator);
    
                    // 次の要素へ
                    query = query_iter.next();
                    doc = doc_iter.next();
                }
                std::cmp::Ordering::Less => {
                    // クエリにしか存在しないトークン
                    query = query_iter.next();
                }
                std::cmp::Ordering::Greater => {
                    // 文書にしか存在しないトークン
                    doc = doc_iter.next();
                }
            }
        }
    
        score
    }
    

    
    
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Document {
    pub text: Option<String>,
    pub tokens: TokenFrequency,
}

impl Document {
    pub fn new() -> Self {
        Document {
            text: None,
            tokens: TokenFrequency::new(),
        }
    }

    pub fn new_with_set(text: Option<&str>, tokens: TokenFrequency) -> Self {
        Document {
            text: text.map(|s| s.to_string()),
            tokens,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentAnalyzer<IdType>
where
    IdType: Eq + std::hash::Hash + Clone + Serialize + Send + Sync,
{
    pub documents: HashMap<IdType, Document>,
    pub idf: TokenFrequency,
    pub total_doc_count: u64,
}

impl<IdType> DocumentAnalyzer<IdType>
where
    IdType: Eq + std::hash::Hash + Clone + Serialize + Send + Sync,
{

    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            idf: TokenFrequency::new(),
            total_doc_count: 0,
        }
    }

    pub fn add_document(&mut self, id: IdType, content: &[&str], text: Option<&str>) -> Option<&Document>{
        if let Some(document) = self.documents.get_mut(&id) {
            self.idf.sub_tokens_string(&document.tokens.get_token_set());
            document.text = text.map(|s| s.to_string());
            document.tokens.reset();
            document. tokens.add_tokens(content);
            self.idf.add_tokens_string(&document.tokens.get_token_set());
            return self.documents.get(&id);
        } else {
            let mut tokens = TokenFrequency::new();
            tokens.add_tokens(content);
            self.idf.add_tokens_string(&tokens.get_token_set());
            self.documents.insert(id.clone(), Document::new_with_set(text, tokens));
            self.total_doc_count += 1;
            return self.documents.get(&id);
        }
    }

    pub fn get_document(&self, id: &IdType) -> Option<&Document> {
        self.documents.get(id)
    }

    pub fn del_document(&mut self, id: &IdType) -> Option<Document> {
        if let Some(document) = self.documents.remove(id) {
            self.total_doc_count -= 1;
            self.idf
                .sub_tokens_string(&document.tokens.get_token_set());
            Some(document)
        } else {
            None
        }
    }

    pub fn get_document_count(&self) -> u64 {
        self.total_doc_count
    }

    pub fn get_token_set_vec(&self) -> Vec<String> {
        self.idf.get_token_set()
    }

    pub fn get_token_set_vec_ref(&self) -> Vec<&str> {
        self.idf.get_token_set_ref()
    }

    pub fn get_token_set(&self) -> HashSet<String> {
        self.idf.get_token_hashset()
    }

    pub fn get_token_set_ref(&self) -> HashSet<&str> {
        self.idf.get_token_hashset_ref()
    }

    pub fn get_token_set_len(&self) -> usize {
        self.idf.get_token_set_len()
    }

    pub fn generate_index(&self) -> Index<IdType> {
        // 統計の初期化
        let total_doc_tokens_len = Arc::new(AtomicU64::new(0));
        let max_doc_tokens_len = Arc::new(AtomicU64::new(0));
        let now_prosessing = Arc::new(AtomicU64::new(0));
    
        // idf のfst生成
        let mut builder = MapBuilder::memory();
        let mut idf_vec = self.idf.get_idf_vector_ref_parallel(self.total_doc_count);
        idf_vec.sort_by(|a, b| a.0.cmp(b.0));
        for (token, idf) in idf_vec {
            builder.insert(token.as_bytes(), idf as u64).unwrap();
        }
        let idf = Arc::new(builder.into_map());
    
        // 並列処理用のスレッドセーフなIndex
        let index = Arc::new(Mutex::new(HashMap::new()));
    
        // ドキュメントごとの処理を並列化
        self.documents.par_iter().for_each(|(id, document)| {
            now_prosessing.fetch_add(1, Ordering::SeqCst);
            println!("{} / {}", now_prosessing.load(Ordering::SeqCst), self.total_doc_count);
            let mut tf_idf_sort_vec: Vec<u16> = Vec::new();
    
            let tf_idf_vec: HashMap<String, u16> =
                document.tokens.get_tfidf_hashmap_fst_parallel(&idf);
    
            let mut stream = idf.stream();
            while let Some((token, _)) = stream.next() {
                let tf_idf = *tf_idf_vec.get(str::from_utf8(token).unwrap()).unwrap_or(&0);
                tf_idf_sort_vec.push(tf_idf);
            }
    
            let tf_idf_csvec: CsVec<u16> = CsVec::from_vec(tf_idf_sort_vec);
            let doc_tokens_len = document.tokens.get_total_token_count();
    
            total_doc_tokens_len.fetch_add(doc_tokens_len, Ordering::SeqCst);
    
            max_doc_tokens_len.fetch_max(doc_tokens_len, Ordering::SeqCst);
    
            let mut index_guard = index.lock().unwrap();
            index_guard.insert(id.clone(), (tf_idf_csvec, doc_tokens_len));
        });
    
        // 統計計算
        let avg_total_doc_tokens_len = (total_doc_tokens_len.load(Ordering::SeqCst)
            / self.total_doc_count as u64) as u64;
        let max_doc_tokens_len = max_doc_tokens_len.load(Ordering::SeqCst);
    
        // indexの返却
        Index::new_with_set(
            Arc::try_unwrap(index).unwrap_or(HashMap::new().into()).into_inner().unwrap(),
            Arc::try_unwrap(idf).unwrap(),
            avg_total_doc_tokens_len,
            max_doc_tokens_len,
            self.total_doc_count,
        )
    }
}