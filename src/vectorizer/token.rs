use core::str;
use std::{collections::{HashMap, HashSet}, fmt::Debug};
use ahash::RandomState;
use serde::{Deserialize, Serialize};


///  TokenFrequency 構造体
/// tokenの出現頻度を管理するための構造体です
/// tokenの出現回数をカウントします。
/// 
/// # Examples
/// ```
/// use vectorizer::token::TokenFrequency;
/// let mut token_freq = TokenFrequency::new();
/// token_freq.add_token("token1");
/// token_freq.add_token("token2");
/// token_freq.add_token("token1");
/// 
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenFrequency {
    token_count: HashMap<String, u64, RandomState>,
    total_token_count: u64,
}

/// Tokenの追加、削除の実装
impl TokenFrequency {
    /// 新しいTokenFrequencyを作成するメソッド
    pub fn new() -> Self {
        TokenFrequency {
            token_count: HashMap::with_hasher(RandomState::new()),
            total_token_count: 0,
        }
    }

    /// tokenを追加する
    /// 
    /// # Arguments
    /// * `token` - 追加するトークン
    #[inline]
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

    /// tokenを引く
    /// 
    /// # Arguments
    /// * `token` - 引くトークン
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

    /// 複数のtokenを引く
    /// 
    /// # Arguments
    /// * `tokens` - 引くトークンのスライス
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

    /// tokenの出現回数を指定する
    /// 
    /// # Arguments
    /// * `token` - トークン
    /// * `count` - 出現回数
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
}

/// TokenFrequencyの情報を取得するための実装
impl TokenFrequency {
    /// すべてのtokenの出現回数を取得します
    /// 
    /// # Returns
    /// * `Vec<(String, u32)>` - トークンとその出現回数のベクタ
    #[inline]
    pub fn token_count_vector(&self) -> Vec<(String, u64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.clone(), count)
        }).collect()
    }

    /// すべてのtokenの出現回数を取得します
    /// 文字列はこれの参照を返します
    /// 
    /// # Returns
    /// * `Vec<(&str, u32)>` - トークンとその出現回数のベクタ
    #[inline]
    pub fn token_count_vector_ref_str(&self) -> Vec<(&str, u64)> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    /// すべてのtokenの出現回数を取得します
    /// 文字列はこれの参照を返します
    /// 
    /// # Returns
    /// * `HashMap<&str, u32>` - トークンとその出現回数のハッシュマップ
    #[inline]
    pub fn token_count_hashmap_ref_str(&self) -> HashMap<&str, u64, RandomState> {
        self.token_count.iter().map(|(token, &count)| {
            (token.as_str(), count)
        }).collect()
    }

    /// 全tokenのカウントの合計を取得します
    /// 
    /// # Returns
    /// * `u64` - tokenのカウントの合計
    #[inline]
    pub fn token_sum(&self) -> u64 {
        self.total_token_count
    }

    /// あるtokenの出現回数を取得します
    /// 
    /// # Arguments
    /// * `token` - トークン
    /// 
    /// # Returns
    /// * `u32` - トークンの出現回数
    #[inline]
    pub fn token_count(&self, token: &str) -> u64 {
        *self.token_count.get(token).unwrap_or(&0)
    }

    /// もっとも多く出現したtokenを取得します
    /// 同じ出現回数のtokenが複数ある場合は、すべてのtokenを取得します
    /// 
    /// # Returns
    /// * `Vec<(String, u32)>` - トークンとその出現回数のベクタ
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

    /// もっとも多く出現したtokenの出現回数取得します
    /// 
    /// # Returns
    /// * `u32` - 再頻出tokenの出現回数
    #[inline]
    pub fn most_frequent_token_count(&self) -> u64 {
        if let Some(&max_count) = self.token_count.values().max() {
            max_count
        } else {
            0
        }
    }

    /// tokenが存在するかどうかを確認します
    /// 
    /// # Arguments
    /// * `token` - トークン
    /// 
    /// # Returns
    /// * `bool` - tokenが存在する場合はtrue、存在しない場合はfalse
    #[inline]
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_count.contains_key(token)
    }

    /// tokenのsetを取得します
    /// 
    /// # Returns
    /// * `Vec<String>` - tokenのset
    #[inline]
    pub fn token_set(&self) -> Vec<String> {
        self.token_count.keys().cloned().collect()
    }

    /// tokenのsetを取得します
    /// 文字列はこれの参照を返します
    /// 
    /// # Returns
    /// * `Vec<&str>` - tokenのset
    #[inline]
    pub fn token_set_ref_str(&self) -> Vec<&str> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    /// tokenのsetを取得します
    /// 
    /// # Returns
    /// * `HashSet<String>` - tokenのset
    #[inline]
    pub fn token_hashset(&self) -> HashSet<String, RandomState> {
        self.token_count.keys().cloned().collect()
    }

    /// tokenのsetを取得します
    /// 文字列はこれの参照を返します
    /// 
    /// # Returns
    /// * `HashSet<&str>` - tokenのset
    #[inline]
    pub fn token_hashset_ref_str(&self) -> HashSet<&str, RandomState> {
        self.token_count.keys().map(|s| s.as_str()).collect()
    }

    /// 出現した単語数を取得します
    /// 
    /// # Returns
    /// * `usize` - 出現した単語数
    #[inline]
    pub fn token_num(&self) -> usize {
        self.token_count.len()
    }

    /// stop_tokenを削除します
    /// 
    /// # Arguments
    /// * `stop_tokens` - 削除するトークンのスライス
    /// 
    /// # Returns
    /// * `u64` - 削除されたtokenの合計数
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

    /// 条件に基づいてtokenを削除します
    /// 
    /// # Arguments
    /// * `condition` - 条件を満たすtokenを削除するクロージャ
    /// 
    /// # Returns
    /// * `u64` - 削除されたtokenの合計数
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

    /// 頻度でソートされたトークンのベクタを取得(降順)
    /// 
    /// # Returns
    /// * `Vec<(String, u32)>` - 頻度でソートされたトークンのベクタ
    #[inline]
    pub fn sorted_frequency_vector(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| b.1.cmp(&a.1));
        token_list
    }

    /// 辞書順でソートされたトークンのベクタを取得(昇順)
    /// 
    /// # Returns
    /// * `Vec<(String, u32)>` - 辞書順でソートされたトークンのベクタ
    #[inline]
    pub fn sorted_dict_order_vector(&self) -> Vec<(String, u64)> {
        let mut token_list: Vec<(String, u64)> = self.token_count
            .iter()
            .map(|(token, &count)| (token.clone(), count))
            .collect();

        token_list.sort_by(|a, b| a.0.cmp(&b.0));
        token_list
    }

    /// tokenの多様性を計算します
    /// 1.0は完全な多様性を示し、0.0は完全な非多様性を示します
    /// 
    /// # Returns
    /// * `f64` - tokenの多様性
    #[inline]
    pub fn unique_token_ratio(&self) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        self.token_count.len() as f64 / self.total_token_count as f64
    }

    /// 確率分布 P(token) を取得します（所有文字列版）
    /// total が 0 の場合は空ベクタを返します
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

    /// 確率分布 P(token) を取得します（&str 参照版）
    /// total が 0 の場合は空ベクタを返します
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

    /// 特定トークンの確率 P(token) を返します
    /// total が 0 の場合は 0.0 を返します
    #[inline]
    pub fn probability(&self, token: &str) -> f64 {
        if self.total_token_count == 0 {
            return 0.0;
        }
        (self.token_count(token) as f64) / (self.total_token_count as f64)
    }

    /// カウントを全リセットします
    #[inline]
    pub fn clear(&mut self) {
        self.token_count.clear();
        self.total_token_count = 0;
    }

    /// shrink internal storage to fit current size
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.token_count.shrink_to_fit();
    }
}