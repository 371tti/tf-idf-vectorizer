use std::collections::{hash_map::Keys, HashMap, HashSet};


pub struct WordFrequency<IdType> {
    pub word_count: HashMap<String, u64>,
    pub total_word_count: u64,
    pub id: IdType,
}

impl<IdType> WordFrequency<IdType> {
    pub fn new(id: IdType) -> Self {
        WordFrequency {
            id: id,
            word_count: HashMap::new(),
            total_word_count: 0,
        }
    }

    pub fn new_with_id(id: IdType) -> Self {
        WordFrequency {
            id: id,
            word_count: HashMap::new(),
            total_word_count: 0,
        }
    }

    pub fn set_id(&mut self, id: IdType) -> &mut Self {
        self.id = id;
        self
    }

    pub fn add_word(&mut self, word: &str) -> &mut Self {
        let count = self.word_count.entry(word.to_string()).or_insert(0);
        *count += 1;
        self.total_word_count += 1;
        self
    }

    pub fn get_tf_vector(&self) -> Vec<(String, f64)> {
        self.word_count.iter().map(|(word, &count)| {
            (word.clone(), count as f64 / self.total_word_count as f64)
        }).collect()
    }

    pub fn get_tf_vector_ref(&self) -> Vec<(&str, f64)> {
        self.word_count.iter().map(|(word, &count)| {
            (word.as_str(), count as f64 / self.total_word_count as f64)
        }).collect()
    }
    

    pub fn get_tf_hashmap(&self) -> HashMap<String, f64> {
        self.word_count.iter().map(|(word, &count)| {
            (word.clone(), count as f64 / self.total_word_count as f64)
        }).collect()
    }

    pub fn get_tf_hashmap_ref(&self) -> HashMap<&str, f64> {
        self.word_count.iter().map(|(word, &count)| {
            (word.as_str(), count as f64 / self.total_word_count as f64)
        }).collect()
    }

    pub fn get_word_tf(&self, word: &str) -> f64 {
        let count = self.word_count.get(word).copied().unwrap_or(0);
        if self.total_word_count == 0 {
            0.0
        } else {
            count as f64 / self.total_word_count as f64
        }
    }
    

    pub fn get_word_count_vector(&self) -> Vec<(String, u64)> {
        self.word_count.iter().map(|(word, &count)| {
            (word.clone(), count)
        }).collect()
    }

    pub fn get_word_count_hashmap(&self) -> HashMap<String, u64> {
        self.word_count.clone()
    }

    pub fn get_total_word_count(&self) -> u64 {
        self.total_word_count
    }

    pub fn get_total_word_count_ref(&self) -> &u64 {
        &self.total_word_count
    }

    pub fn get_id(&self) -> &IdType {
        &self.id
    }

    pub fn get_word_count(&self, word: &str) -> u64 {
        *self.word_count.get(word).unwrap_or(&0)
    }

    pub fn get_word_count_ref(&self, word: &str) -> &u64 {
        self.word_count.get(word).unwrap_or(&0)
    }

    pub fn get_most_frequent_word(&self) -> Option<(&String, &u64)> {
        self.word_count.iter().max_by_key(|&(_, count)| count)
    }

    pub fn get_tfidf_vector(&self, idf_map: &HashMap<String, f64>) -> Vec<(String, f64)> {
        self.word_count.iter().map(|(word, &count)| {
            let tf = count as f64 / self.total_word_count as f64;
            let idf = idf_map.get(word).copied().unwrap_or(0.0);
            (word.clone(), tf * idf)
        }).collect()
    }

    pub fn contains_word(&self, word: &str) -> bool {
        self.word_count.contains_key(word)
    }

    pub fn get_word_set(&self) -> Vec<String> {
        self.word_count.keys().cloned().collect()
    }

    pub fn get_word_set_ref(&self) -> Vec<&str> {
        self.word_count.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_word_hashset(&self) -> HashSet<String> {
        self.word_count.keys().cloned().collect()
    }

    pub fn get_word_hashset_ref(&self) -> HashSet<&str> {
        self.word_count.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_word_set_len(&self) -> usize {
        self.word_count.len()
    }

    pub fn get_word_set_iter(&self) -> Keys<String, u64> {
        self.word_count.keys()
    }
    
    pub fn get_word_set_iter_ref(&self) -> impl Iterator<Item = &str> {
        self.word_count.keys().map(|s| s.as_str())
    }

    pub fn get_word_length_stats(&self) -> Option<(usize, usize, f64)> {
        if self.word_count.is_empty() {
            return None;
        }
    
        let lengths: Vec<usize> = self.word_count.keys().map(|word| word.len()).collect();
        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let avg_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    
        Some((min_len, max_len, avg_len))
    }

    pub fn get_word_length_stats_ref(&self) -> Option<(usize, usize, f64)> {
        if self.word_count.is_empty() {
            return None;
        }
    
        let lengths: Vec<usize> = self.word_count.keys().map(|word| word.len()).collect();
        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        let avg_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    
        Some((min_len, max_len, avg_len))
    }

    pub fn remove_stop_words(&mut self, stop_words: &[&str]) {
        for &stop_word in stop_words {
            if let Some(count) = self.word_count.remove(stop_word) {
                self.total_word_count -= count; // total_word_count から引く
            }
        }
    }

    /// 出現回数の降順でソート（多い順）
    pub fn get_sorted_by_frequency_desc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| b.1.cmp(&a.1)); // 降順ソート
        word_list
    }

    /// 出現回数の昇順でソート（少ない順）
    pub fn get_sorted_by_frequency_asc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| a.1.cmp(&b.1)); // 昇順ソート
        word_list
    }

    /// アルファベット順（昇順）でソート
    pub fn get_sorted_by_alphabetical_asc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| a.0.cmp(&b.0)); // アルファベット昇順
        word_list
    }

    /// アルファベット順（降順）でソート
    pub fn get_sorted_by_alphabetical_desc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| b.0.cmp(&a.0)); // アルファベット降順
        word_list
    }

    /// 単語の長さの降順でソート（長い順）
    pub fn get_sorted_by_length_desc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // 長さの降順
        word_list
    }

    /// 単語の長さの昇順でソート（短い順）
    pub fn get_sorted_by_length_asc(&self) -> Vec<(String, u64)> {
        let mut word_list: Vec<(String, u64)> = self.word_count
            .iter()
            .map(|(word, &count)| (word.clone(), count))
            .collect();

        word_list.sort_by(|a, b| a.0.len().cmp(&b.0.len())); // 長さの昇順
        word_list
    }

    pub fn get_unique_word_ratio(&self) -> f64 {
        if self.total_word_count == 0 {
            return 0.0;
        }
        self.word_count.len() as f64 / self.total_word_count as f64
    }
    
}

pub struct TF {

}