use std::sync::atomic::{AtomicU64, Ordering};
use ahash::RandomState;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// keep document count and token counts in a thread-safe way
#[derive(Debug, Serialize, Deserialize)]
pub struct Corpus {
    /// corpus add_num
    /// for update notify
    pub add_num: AtomicU64,
    /// clone以外はRelaxedで十分だと思う
    pub sub_num: AtomicU64,
    // ハッシュ関数は ahash に変更
    pub token_counts: DashMap<String, u64, RandomState>,
}

impl Clone for Corpus {
    fn clone(&self) -> Self {
        Self {
            add_num: AtomicU64::new(self.add_num.load(Ordering::Acquire)),
            sub_num: AtomicU64::new(self.sub_num.load(Ordering::Acquire)),
            token_counts: self.token_counts.clone(),
        }
    }
}

impl Corpus {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            add_num: AtomicU64::new(0),
            sub_num: AtomicU64::new(0),
            token_counts: DashMap::with_hasher(RandomState::new()),
        }
    }

    /// Add a document's tokens to the corpus
    pub fn add_doc(&self, tokens: &[&str]) {
        self.add_num.fetch_add(1, Ordering::Relaxed);
        for token in tokens {
            self.token_counts
                .entry(token.to_string())
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    pub fn remove_doc(&self, tokens: &[&str]) {
        self.sub_num.fetch_add(1, Ordering::Relaxed);
        for token in tokens {
            if let Some(mut count) = self.token_counts.get_mut(*token) {
                if *count > 1 {
                    *count -= 1;
                } else {
                    self.token_counts.remove(*token);
                }
            }
        }
    }

    /// Get the number of documents in the corpus
    pub fn get_doc_num(&self) -> u64 {
        let add_num = self.add_num.load(Ordering::Relaxed);
        let sub_num = self.sub_num.load(Ordering::Relaxed);
        add_num - sub_num
    }

    /// Get the generation number of the corpus
    pub fn get_gen_num(&self) -> u64 {
        let add_num = self.add_num.load(Ordering::Relaxed);
        let sub_num = self.sub_num.load(Ordering::Relaxed);
        add_num + sub_num
    }

    /// corpus上のtokenの出現回数を取得する
    pub fn get_token_count(&self, token: &str) -> u64 {
        self.token_counts.get(token).map_or(0, |count| *count)
    }
}
