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
    /// corpus sub_num
    /// for update notify
    pub sub_num: AtomicU64,
    // token counts in corpus
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
    pub fn add_set<T>(&self, tokens: &[T])
    where
        T: AsRef<str>,
    {
        self.add_num.fetch_add(1, Ordering::Relaxed);
        for token in tokens {
            self.token_counts
                .entry(token.as_ref().to_string())
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    pub fn sub_set<T>(&self, tokens: &[T])
    where
        T: AsRef<str>,
    {
        self.sub_num.fetch_add(1, Ordering::Relaxed);
        for token in tokens {
            if let Some(mut count) = self.token_counts.get_mut(token.as_ref()) {
                if *count > 1 {
                    *count -= 1;
                } else {
                    drop(count);
                    self.token_counts.remove(token.as_ref());
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

    /// Get the token count in the corpus
    pub fn get_token_count(&self, token: &str) -> u64 {
        self.token_counts.get(token).map_or(0, |count| *count)
    }

    /// Get the current vocabulary size (number of unique tokens)
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.token_counts.len()
    }
}
