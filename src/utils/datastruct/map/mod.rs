use hashbrown::HashMap;
use std::hash::BuildHasher;
use std::hash::Hasher;

pub struct KeyIndexMap<K, V> {
    pub entries: Vec<Entry<K, V>>,
    pub table: HashMap<K, usize, ahash::RandomState>,
}

pub struct Entry<K, V> {
    pub key: K,
    pub value: V,
}

pub struct HashKey {
    pub raw_hash: u64,
}

impl<K, V> KeyIndexMap<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            table: HashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    pub fn key_hash(&self, key: &K) -> u64 {
        let mut hasher = self.table.hasher().build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.table.get(key).map(|&idx| &self.entries[idx].value)
    }

    pub fn get_with_hash_key(&self, hash_key: &HashKey) -> Option<&V> {
        self.table.raw_entry().from_hash(hash, is_match)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&idx) = self.table.get(&key) {
            let old_value = std::mem::replace(&mut self.entries[idx].value, value);
            Some(old_value)
        } else {
            let idx = self.entries.len();
            self.entries.push(Entry { key: key.clone(), value });
            self.table.insert(key, idx);
            None
        }
    }
}