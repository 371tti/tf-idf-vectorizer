use std::collections::{hash_map::Keys, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use rayon::{prelude::*, vec};
// use trie_rs::map::TrieBuilder;

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

    pub fn idf_max(&self, total_doc_count: u64) -> f64 {
        if let Some(&max_count) = self.token_count.values().max() {
            (total_doc_count as f64 / (1.0 + self.get_most_frequent_token_count() as f64)).ln()
        } else {
            0.0
        }
    }

    pub fn idf_calc(total_doc_count: u64, max_idf: f64, doc_count: u32) -> f64 {
        (total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf
    }

    pub fn idf_calc_as_u16(total_doc_count: u64, max_idf: f64, doc_count: u32) -> u16 {
        let normalized_value = ((total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf);
        // 0～65535 にスケール
        (normalized_value * 65535.0).round() as u16
    }

    pub fn idf_calc_as_u32(total_doc_count: u64, max_idf: f64, doc_count: u32) -> u32 {
        let normalized_value = ((total_doc_count as f64 / (1.0 + doc_count as f64)).ln() / max_idf);
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

    pub fn get_tfidf_vector(&self, idf_map: &HashMap<String, u16>) -> Vec<(String, u16)> {
        self.token_count.iter().map(|(token, &count)| {
            let tf = Self::tf_calc_as_u16(self.get_most_frequent_token_count(), count);
            let idf = idf_map.get(token).copied().unwrap_or(0);
            (token.clone(), ((tf as u32 * idf as u32) / u16::MAX as u32) as u16)
        }).collect()
    }

    pub fn get_tfidf_vector_parallel(&self, idf_map: &HashMap<String, u16>) -> Vec<(String, f64)> {
        self.token_count
            .par_iter()
            .map(|(token, &count)| {
                let tf = Self::tf_calc(self.get_most_frequent_token_count(), count);
                let idf = idf_map.get(token).copied().unwrap_or(0) as f64;
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
}







// pub struct Index<IdType> {
//     pub index: HashMap<IdType, Vec<u16>>,
//     pub index_float_max: (f64, f64),
//     pub idf: IndexMap<String, (u16, u32)>,
//     pub idf_float_max: f64,
//     pub total_doc_count: u64,
// }


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Document {
    pub text: String,
    pub tokens: TokenFrequency,
}

impl Document {
    pub fn new() -> Self {
        Document {
            text: String::new(),
            tokens: TokenFrequency::new(),
        }
    }

    pub fn new_with_set(text: &str, tokens: TokenFrequency) -> Self {
        Document {
            text: text.to_string(),
            tokens,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentAnalyzer<IdType, Splitter>
where
    IdType: Eq + std::hash::Hash,
    Splitter: Fn(&str) -> Vec<String>,
{
    pub documents: HashMap<IdType, Document>,
    pub idf: TokenFrequency,
    pub total_doc_count: u64,
    pub spliter: Splitter,
}

impl<IdType, Splitter> DocumentAnalyzer<IdType, Splitter>
where
    IdType: Eq + std::hash::Hash,
    Splitter: Fn(&str) -> Vec<String>,
{

    pub fn new_with_spliter(splitter: Splitter) -> Self {
        Self {
            documents: HashMap::new(),
            idf: TokenFrequency::new(),
            total_doc_count: 0,
            spliter: splitter,
        }
    }

    pub fn add_document(&mut self, id: IdType, content: &str) {
        let binding = (self.spliter)(content);
        let mut token_frequency = TokenFrequency::new();
        token_frequency.add_tokens_string(&binding);
        self.idf
        .add_tokens(&token_frequency.get_token_set_ref());
        self.documents
        .insert(id, Document::new_with_set(content, token_frequency));
        self.total_doc_count += 1;
    }

    pub fn get_document(&self, id: &IdType) -> Option<&Document> {
        self.documents.get(id)
    }

    pub fn del_document(&mut self, id: &IdType) -> Option<Document> {
        if let Some(document) = self.documents.remove(id) {
            self.total_doc_count -= 1;
            self.idf
                .sub_tokens(&document.tokens.get_token_set_ref());
            Some(document)
        } else {
            None
        }
    }

    // pub fn generate_index(&self) -> Index<IdType> {
    //     let mut index: HashMap<IdType, Vec<u16>> = HashMap::new();
    //     let mut idf = self.idf.get_idf_vector_ref_parallel(self.total_doc_count);
    //     for (id, document) in self.documents.iter() {
    //         let mut tf_idf_vec: Vec<f64> = Vec::new();
    //         for (token, idf) in idf.iter() {
    //             let tf = document.tokens.get_token_tf(token);
    //             let tf_idf = tf * idf;
    //             tf_idf_vec.push(tf_idf);
    //         }
            
    //     }
    //     let mut index_float_max = (0.0, 0.0);
    //     let mut idf_float_max = 0.0;
    //     let doc = self.documents.iter()
    // }
}