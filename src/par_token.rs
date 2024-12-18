use std::collections::{hash_map::Keys, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency<IdType> {
    pub token_count: HashMap<String, u64>,
    pub total_token_count: u64,
    pub id: IdType,
}

impl<IdType: Send + Sync> TokenFrequency<IdType> {
    pub fn new(id: IdType) -> Self {
        TokenFrequency {
            id,
            token_count: HashMap::new(),
            total_token_count: 0,
        }
    }

    pub fn new_with_id(id: IdType) -> Self {
        TokenFrequency {
            id,
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

    /// 並列: TFベクトルの取得
    pub fn get_tf_vector_parallel(&self) -> Vec<(String, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    /// 並列: TFベクトル(参照)の取得
    pub fn get_tf_vector_ref_parallel(&self) -> Vec<(&str, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.as_str(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    /// 並列: TFハッシュマップの取得
    pub fn get_tf_hashmap_parallel(&self) -> HashMap<String, f64> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count as f64 / self.total_token_count as f64))
            .collect()
    }

    /// 並列: TFハッシュマップ(参照)の取得
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

    /// 並列: トークンカウントベクタの取得
    pub fn get_token_count_vector_parallel(&self) -> Vec<(String, u64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect()
    }

    pub fn get_token_count_hashmap(&self) -> HashMap<String, u64> {
        self.token_count.clone()
    }

    pub fn get_total_token_count(&self) -> u64 {
        self.total_token_count
    }

    pub fn get_id(&self) -> &IdType {
        &self.id
    }

    pub fn get_token_count(&self, token: &str) -> u64 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    /// 並列: 最頻出トークン取得
    pub fn get_most_frequent_token_parallel(&self) -> Option<(String, u64)> {
        self.token_count
            .par_iter()
            .reduce_with(|a, b| if a.1 >= b.1 { a } else { b })
            .map(|(token, &count)| (token.clone(), count))
    }

    /// 並列: TF-IDFベクトル取得
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

    /// 並列: トークンセット取得
    pub fn get_token_set_parallel(&self) -> Vec<String> {
        self.token_count
            .par_iter()
            .map(|(token, _)| token.clone())
            .collect()
    }

    /// 並列: トークンセット(参照)取得
    pub fn get_token_set_ref_parallel(&self) -> Vec<&str> {
        self.token_count
            .par_iter()
            .map(|(token, _)| token.as_str())
            .collect()
    }

    /// 並列: トークンハッシュセット取得
    pub fn get_token_hashset_parallel(&self) -> HashSet<String> {
        self.token_count
            .par_iter()
            .map(|(token, _)| token.clone())
            .collect()
    }

    /// 並列: トークンハッシュセット(参照)取得
    pub fn get_token_hashset_ref_parallel(&self) -> HashSet<&str> {
        self.token_count
            .par_iter()
            .map(|(token, _)| token.as_str())
            .collect()
    }

    pub fn get_token_set_len(&self) -> usize {
        self.token_count.len()
    }

// 並列処理でトークン長の統計情報を計算
pub fn get_token_length_stats_parallel(&self) -> Option<(usize, usize, f64)> {
    if self.token_count.is_empty() {
        return None;
    }

    // reduce を用いて並列で集計
    let (min_len, max_len, total_len, count) = self.token_count
        .par_iter()
        .map(|(token, _)| (token.len(), token.len(), token.len(), 1)) // 初期値: (min, max, sum, count)
        .reduce(
            || (usize::MAX, 0, 0, 0), // 結合時の初期値
            |acc, len| {
                // acc と len をマージ
                let min_len = acc.0.min(len.0);
                let max_len = acc.1.max(len.1);
                let total_len = acc.2 + len.2;
                let count = acc.3 + len.3;
                (min_len, max_len, total_len, count)
            },
        );

    // 結果を Some で返す
    Some((min_len, max_len, total_len as f64 / count as f64))
}

    pub fn remove_stop_tokens(&mut self, stop_tokens: &[&str]) {
        for &stop_token in stop_tokens {
            if let Some(count) = self.token_count.remove(stop_token) {
                self.total_token_count -= count;
            }
        }
    }

    pub fn remove_tokens_by_condition<F>(&mut self, condition: F) -> u64
    where
        F: Fn(&str, &u64) -> bool + Send + Sync,
    {
        let mut removed_total_count = 0;
        // retain は並列化しづらいのでそのまま
        self.token_count.retain(|token, count| {
            if condition(token, count) {
                removed_total_count += *count;
                false
            } else {
                true
            }
        });
        self.total_token_count -= removed_total_count;

        removed_total_count
    }

    /// 並列: 出現回数の降順でソート
    pub fn get_sorted_by_frequency_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    /// 並列: 出現回数の昇順でソート
    pub fn get_sorted_by_frequency_asc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.1.cmp(&b.1));
        token_list
    }

    /// 並列: アルファベット順（昇順）でソート
    pub fn get_sorted_by_alphabetical_asc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    /// 並列: アルファベット順（降順）でソート
    pub fn get_sorted_by_alphabetical_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.cmp(&a.0));
        token_list
    }

    /// 並列: 単語の長さの降順でソート
    pub fn get_sorted_by_length_desc_parallel(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .par_iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.par_sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        token_list
    }

    /// 並列: 単語の長さの昇順でソート
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
