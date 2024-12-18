use std::collections::{hash_map::Keys, HashMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency<IdType> {
    pub token_count: HashMap<String, u64>,
    pub total_token_count: u64,
    pub id: IdType,
}

impl<IdType> TokenFrequency<IdType> {
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

    pub fn get_tf_vector_ref(&self) -> Vec<(&str, f64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count as f64 / self.total_token_count as f64)
        }).collect()
    }
    

    pub fn get_tf_hashmap(&self) -> HashMap<String, f64> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count as f64 / self.total_token_count as f64)
        }).collect()
    }

    pub fn get_tf_hashmap_ref(&self) -> HashMap<&str, f64> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count as f64 / self.total_token_count as f64)
        }).collect()
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

    pub fn get_most_frequent_token(&self) -> Option<(&String, &u64)> {
        self.token_count.iter().max_by_key(|&(_, count)| count)
    }

    pub fn get_tfidf_vector(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = count as f64 / self.total_token_count as f64;
            let idf = idf_map.get(token).copied().unwrap_or(0.0);
            (token.clone(), tf * idf)
        }).collect()
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

    pub fn remove_stop_tokens(&mut self, stop_tokens: &[&str]) {
        for &stop_token in stop_tokens {
            if let Some(count) = self.token_count.remove(stop_token) {
                self.total_token_count -= count; // total_token_count から引く
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
                false // 削除する
            } else {
                true // 残す
            }
        });
        self.total_token_count -= removed_total_count;
    
        removed_total_count
    }

    /// 出現回数の降順でソート（多い順）
    pub fn get_sorted_by_frequency_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1)); // 降順ソート
        token_list
    }

    /// 出現回数の昇順でソート（少ない順）
    pub fn get_sorted_by_frequency_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.1.cmp(&b.1)); // 昇順ソート
        token_list
    }

    /// アルファベット順（昇順）でソート
    pub fn get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0)); // アルファベット昇順
        token_list
    }

    /// アルファベット順（降順）でソート
    pub fn get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.cmp(&a.0)); // アルファベット降順
        token_list
    }

    /// 単語の長さの降順でソート（長い順）
    pub fn get_sorted_by_length_desc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // 長さの降順
        token_list
    }

    /// 単語の長さの昇順でソート（短い順）
    pub fn get_sorted_by_length_asc(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.len().cmp(&b.0.len())); // 長さの昇順
        token_list
    }

    pub fn get_unique_token_ratio(&self) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        self.token_count.len() as f64 / self.total_token_count as f64
    }
    

}

