use hashbrown::HashMap;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;
use std::rc::Weak;

pub mod serde;

pub struct KeyIndexMap<K, V> {
    pub entries: Vec<Entry<KeyRc<K>, V>>,
    pub table: HashMap<KeyRc<K>, usize, ahash::RandomState>,
}

#[derive(Debug)]
pub struct KeyRc<K> {
    pub rc: Rc<K>,
}

impl<K> Clone for KeyRc<K> {
    fn clone(&self) -> Self {
        KeyRc {
            rc: self.rc.clone(),
        }
    }
}

#[derive(Debug)]
pub struct WeakKey<K> {
    pub weak: Weak<K>,
}

impl<K> Clone for WeakKey<K> {
    fn clone(&self) -> Self {
        WeakKey {
            weak: self.weak.clone(),
        }
    }
}

impl<K> KeyRc<K> {
    pub fn new(key: K) -> Self {
        KeyRc {
            rc: Rc::new(key),
        }
    }

    pub fn to_weak(&self) -> WeakKey<K> {
        WeakKey {
            weak: Rc::downgrade(&self.rc),
        }
    }
}

impl<K> AsRef<K> for KeyRc<K> {
    fn as_ref(&self) -> &K {
        self.rc.as_ref()
    }
}

impl<K> Hash for KeyRc<K> 
where K: Hash
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // ptr先実態のHashを呼び出すのだよ
        self.rc.as_ref().hash(state);
    }
}

impl<K> Borrow<K> for KeyRc<K> {
    fn borrow(&self) -> &K {
        self.rc.as_ref()
    }
}

impl<K> PartialEq for KeyRc<K> 
where K: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.rc.as_ref().eq(other.rc.as_ref())
    }
}
impl<K> Eq for KeyRc<K> 
where K: Eq {}

#[derive(Clone)]
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
        KeyIndexMap {
            entries: Vec::new(),
            table: HashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.table.get(key) {
            unsafe {
                Some(&self.entries.get_unchecked(idx).value)
            }
        } else {
            None
        }
    }

    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        if let Some(&idx) = self.table.get(key) {
            unsafe {
                let entry = self.entries.get_unchecked(idx);
                Some((entry.key.rc.as_ref(), &entry.value))
            }
        } else {
            None
        }
    }

    pub fn get_with_weak_key(&self, key: &WeakKey<K>) -> Option<&V> {
        if let Some(strong_key) = key.weak.upgrade() {
            if let Some(&idx) = self.table.get(strong_key.as_ref()) {
                unsafe {
                    Some(&self.entries.get_unchecked(idx).value)
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> InsertResult<K, V> {
        let key_rc = KeyRc::new(key);
        self.insert_with_key_rc(key_rc, value)
    }

    pub fn insert_with_key_rc(&mut self, key_rc: KeyRc<K>, value: V) -> InsertResult<K, V> {
        if let Some((key, &idx)) = self.table.get_key_value(&key_rc) {
            // Key exists, update value
            let old_value = Some(self.entries[idx].value.clone());
            self.entries[idx].value = value;
            InsertResult {
                old_value,
                weak_key: key.to_weak(),
            }
            // drop key_rc
        } else {
            // New key, insert entry
            let idx = self.entries.len();
            self.entries.push(Entry {
                key: key_rc.clone(),
                value,
            });
            self.table.insert(key_rc.clone(), idx);
            InsertResult {
                old_value: None,
                weak_key: key_rc.to_weak(),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.table.contains_key(key)
    }

    pub fn swap_remove(&mut self, key: &K) -> Option<V> {
        if let Some(&idx) = self.table.get(key) {
            let entry = self.entries.swap_remove(idx);
            self.table.remove(key);
            // swapで移動させられた要素のインデックスを更新
            let moved_entry = &self.entries[idx];
            self.table.insert(moved_entry.key.clone(), idx);
            Some(entry.value)
        } else {
            None
        }
    }

    pub fn iter(&self) -> KeyIndexMapIter<K, V> {
        KeyIndexMapIter {
            map: self,
            index: 0,
        }
    }

    pub fn raw_iter(&self) -> RawKeyIndexMapIter<K, V> {
        RawKeyIndexMapIter {
            map: self,
            index: 0,
        }
    }
}

pub struct KeyIndexMapIter<'a, K, V> {
    pub map: &'a KeyIndexMap<K, V>,
    pub index: usize,
}

impl<'a, K, V> Iterator for KeyIndexMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.map.entries.len() {
            None
        } else {
            let entry = &self.map.entries[self.index];
            self.index += 1;
            Some((entry.key.rc.as_ref(), &entry.value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.map.entries.len() - self.index;
        (len, Some(len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;
        self.next()
    }
}

pub struct RawKeyIndexMapIter<'a, K, V> {
    pub map: &'a KeyIndexMap<K, V>,
    pub index: usize,
}

impl<'a, K, V> Iterator for RawKeyIndexMapIter<'a, K, V> {
    type Item = (&'a KeyRc<K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.map.entries.len() {
            None
        } else {
            let entry = &self.map.entries[self.index];
            self.index += 1;
            Some((&entry.key, &entry.value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.map.entries.len() - self.index;
        (len, Some(len))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;
        self.next()
    }
}

impl<K, V> Debug for KeyIndexMap<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|entry| (entry.key.rc.as_ref(), &entry.value)))
            .finish()
    }
}

impl<K, V> Clone for KeyIndexMap<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Rcに関してはそのままRcとしてクローンされる
    /// 問題あればDeep Cloneを実装すること
    fn clone(&self) -> Self {
        KeyIndexMap {
            entries: self.entries.clone(),
            table: self.table.clone(),
        }
    }
}

pub struct InsertResult<K, V> {
    pub old_value: Option<V>,
    pub weak_key: WeakKey<K>,
}