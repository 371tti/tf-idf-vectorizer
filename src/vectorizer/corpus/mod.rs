use std::sync::atomic::{AtomicU64, Ordering};
use ahash::RandomState;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::TermFrequency;

/// keep document count and term counts in a thread-safe way
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Corpus {
    /// corpus add_num
    /// for update notify
    pub add_num: AtomicU64,
    /// corpus sub_num
    /// for update notify
    pub sub_num: AtomicU64,
    // term counts in corpus
    pub term_counts: DashMap<Box<str>, u64, RandomState>,
}

impl Clone for Corpus {
    fn clone(&self) -> Self {
        Self {
            add_num: AtomicU64::new(self.add_num.load(Ordering::Acquire)),
            sub_num: AtomicU64::new(self.sub_num.load(Ordering::Acquire)),
            term_counts: self.term_counts.clone(),
        }
    }
}

impl Corpus {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            add_num: AtomicU64::new(0),
            sub_num: AtomicU64::new(0),
            term_counts: DashMap::with_hasher(RandomState::new()),
        }
    }

    /// Add a document's terms to the corpus
    pub fn add_set<T>(&self, terms: &[T])
    where
        T: AsRef<str>,
    {
        self.add_num.fetch_add(1, Ordering::Relaxed);
        for term in terms {
            self.term_counts
                .entry(term.as_ref().into())
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    pub fn sub_set<T>(&self, terms: &[T])
    where
        T: AsRef<str>,
    {
        self.sub_num.fetch_add(1, Ordering::Relaxed);
        for term in terms {
            if let Some(mut count) = self.term_counts.get_mut(term.as_ref()) {
                if *count > 1 {
                    *count -= 1;
                } else {
                    drop(count);
                    self.term_counts.remove(term.as_ref());
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

    /// Get the term count in the corpus
    pub fn get_term_count(&self, term: &str) -> u64 {
        self.term_counts.get(term).map_or(0, |count| *count)
    }

    /// Get the current vocabulary size (number of unique terms)
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.term_counts.len()
    }
}

/// for analyze usage
impl Corpus {
    /// Get all terms in the corpus
    pub fn get_all_terms(&self) -> Vec<String> {
        self.term_counts
            .iter()
            .map(|entry| entry.key().to_string())
            .collect()
    }

    /// self - other
    /// for trend analyze
    pub fn delta_corpus(&self, other: &Corpus) -> Corpus {
        let delta = Corpus::new();
        for entry in self.term_counts.iter() {
            let term = entry.key();
            let count_self = *entry.value();
            let count_other = other.term_counts.get(term).map_or(0, |v| *v);
            if count_self > count_other {
                delta
                    .term_counts
                    .insert(term.clone(), count_self - count_other);
            }
        }
        delta.add_num
            .store(self.add_num.load(Ordering::Relaxed), Ordering::Relaxed);
        delta.sub_num
            .store(self.sub_num.load(Ordering::Relaxed), Ordering::Relaxed);
        delta
    }

    /// Merge another corpus into self
    pub fn merge_corpus(&self, other: &Corpus) {
        for entry in other.term_counts.iter() {
            let term = entry.key();
            let count_other = *entry.value();
            self.term_counts
                .entry(term.clone())
                .and_modify(|count| *count += count_other)
                .or_insert(count_other);
        }
        self.add_num
            .fetch_add(other.add_num.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sub_num
            .fetch_add(other.sub_num.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

impl Into<TermFrequency> for &Corpus {
    fn into(self) -> TermFrequency {
        let mut tf = TermFrequency::new();
        for entry in self.term_counts.iter() {
            let term = entry.key();
            let count = *entry.value();
            tf.set_term_count(term, count);
        }
        tf
    }
}