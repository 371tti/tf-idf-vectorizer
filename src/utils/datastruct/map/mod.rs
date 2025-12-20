use hashbrown::HashMap;
use hashbrown::HashTable;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr;
use std::ptr::NonNull;
use std::rc::Rc;
use std::rc::Weak;

// pub mod serde;

// pub struct KeyIndexMap<K, V> {
//     pub entries: Vec<Entry<KeyRc<K>, V>>,
//     pub table: HashMap<KeyRc<K>, usize, ahash::RandomState>,
// }

// #[derive(Debug)]
// pub struct KeyRc<K> {
//     pub rc: Rc<K>,
// }

// impl<K> Clone for KeyRc<K> {
//     fn clone(&self) -> Self {
//         KeyRc {
//             rc: self.rc.clone(),
//         }
//     }
// }

// #[derive(Debug)]
// pub struct WeakKey<K> {
//     pub weak: Weak<K>,
// }

// impl<K> Clone for WeakKey<K> {
//     fn clone(&self) -> Self {
//         WeakKey {
//             weak: self.weak.clone(),
//         }
//     }
// }

// impl<K> KeyRc<K> {
//     pub fn new(key: K) -> Self {
//         KeyRc {
//             rc: Rc::new(key),
//         }
//     }

//     pub fn to_weak(&self) -> WeakKey<K> {
//         WeakKey {
//             weak: Rc::downgrade(&self.rc),
//         }
//     }
// }

// impl<K> AsRef<K> for KeyRc<K> {
//     fn as_ref(&self) -> &K {
//         self.rc.as_ref()
//     }
// }

// impl<K> Hash for KeyRc<K> 
// where K: Hash
// {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         // ptr先実態のHashを呼び出すのだよ
//         self.rc.as_ref().hash(state);
//     }
// }

// impl<K> Borrow<K> for KeyRc<K> {
//     fn borrow(&self) -> &K {
//         self.rc.as_ref()
//     }
// }

// impl<K> PartialEq for KeyRc<K> 
// where K: PartialEq
// {
//     fn eq(&self, other: &Self) -> bool {
//         self.rc.as_ref().eq(other.rc.as_ref())
//     }
// }
// impl<K> Eq for KeyRc<K> 
// where K: Eq {}

// #[derive(Clone)]
// pub struct Entry<K, V> {
//     pub key: K,
//     pub value: V,
// }

// pub struct HashKey {
//     pub raw_hash: u64,
// }

// impl<K, V> KeyIndexMap<K, V>
// where
//     K: Eq + std::hash::Hash + Clone,
//     V: Clone,
// {
//     pub fn new() -> Self {
//         KeyIndexMap {
//             entries: Vec::new(),
//             table: HashMap::with_hasher(ahash::RandomState::new()),
//         }
//     }

//     pub fn get(&self, key: &K) -> Option<&V> {
//         if let Some(&idx) = self.table.get(key) {
//             unsafe {
//                 Some(&self.entries.get_unchecked(idx).value)
//             }
//         } else {
//             None
//         }
//     }

//     pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
//         if let Some(&idx) = self.table.get(key) {
//             unsafe {
//                 let entry = self.entries.get_unchecked(idx);
//                 Some((entry.key.rc.as_ref(), &entry.value))
//             }
//         } else {
//             None
//         }
//     }

//     pub fn get_with_weak_key(&self, key: &WeakKey<K>) -> Option<&V> {
//         if let Some(strong_key) = key.weak.upgrade() {
//             if let Some(&idx) = self.table.get(strong_key.as_ref()) {
//                 unsafe {
//                     Some(&self.entries.get_unchecked(idx).value)
//                 }
//             } else {
//                 None
//             }
//         } else {
//             None
//         }
//     }

//     pub fn insert(&mut self, key: K, value: V) -> InsertResult<K, V> {
//         let key_rc = KeyRc::new(key);
//         self.insert_with_key_rc(key_rc, value)
//     }

//     pub fn insert_with_key_rc(&mut self, key_rc: KeyRc<K>, value: V) -> InsertResult<K, V> {
//         if let Some((key, &idx)) = self.table.get_key_value(&key_rc) {
//             // Key exists, update value
//             let old_value = Some(self.entries[idx].value.clone());
//             self.entries[idx].value = value;
//             InsertResult {
//                 old_value,
//                 weak_key: key.to_weak(),
//             }
//             // drop key_rc
//         } else {
//             // New key, insert entry
//             let idx = self.entries.len();
//             self.entries.push(Entry {
//                 key: key_rc.clone(),
//                 value,
//             });
//             self.table.insert(key_rc.clone(), idx);
//             InsertResult {
//                 old_value: None,
//                 weak_key: key_rc.to_weak(),
//             }
//         }
//     }

//     pub fn len(&self) -> usize {
//         self.entries.len()
//     }

//     pub fn contains_key(&self, key: &K) -> bool {
//         self.table.contains_key(key)
//     }

//     pub fn swap_remove(&mut self, key: &K) -> Option<V> {
//         if let Some(&idx) = self.table.get(key) {
//             let entry = self.entries.swap_remove(idx);
//             self.table.remove(key);
//             // swapで移動させられた要素のインデックスを更新
//             let moved_entry = &self.entries[idx];
//             self.table.insert(moved_entry.key.clone(), idx);
//             Some(entry.value)
//         } else {
//             None
//         }
//     }

//     pub fn iter(&self) -> KeyIndexMapIter<K, V> {
//         KeyIndexMapIter {
//             map: self,
//             index: 0,
//         }
//     }

//     pub fn raw_iter(&self) -> RawKeyIndexMapIter<K, V> {
//         RawKeyIndexMapIter {
//             map: self,
//             index: 0,
//         }
//     }
// }

// pub struct KeyIndexMapIter<'a, K, V> {
//     pub map: &'a KeyIndexMap<K, V>,
//     pub index: usize,
// }

// impl<'a, K, V> Iterator for KeyIndexMapIter<'a, K, V> {
//     type Item = (&'a K, &'a V);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.map.entries.len() {
//             None
//         } else {
//             let entry = &self.map.entries[self.index];
//             self.index += 1;
//             Some((entry.key.rc.as_ref(), &entry.value))
//         }
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.map.entries.len() - self.index;
//         (len, Some(len))
//     }

//     fn nth(&mut self, n: usize) -> Option<Self::Item> {
//         self.index += n;
//         self.next()
//     }
// }

// pub struct RawKeyIndexMapIter<'a, K, V> {
//     pub map: &'a KeyIndexMap<K, V>,
//     pub index: usize,
// }

// impl<'a, K, V> Iterator for RawKeyIndexMapIter<'a, K, V> {
//     type Item = (&'a KeyRc<K>, &'a V);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.map.entries.len() {
//             None
//         } else {
//             let entry = &self.map.entries[self.index];
//             self.index += 1;
//             Some((&entry.key, &entry.value))
//         }
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.map.entries.len() - self.index;
//         (len, Some(len))
//     }

//     fn nth(&mut self, n: usize) -> Option<Self::Item> {
//         self.index += n;
//         self.next()
//     }
// }

// impl<K, V> Debug for KeyIndexMap<K, V>
// where
//     K: Debug,
//     V: Debug,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_map()
//             .entries(self.entries.iter().map(|entry| (entry.key.rc.as_ref(), &entry.value)))
//             .finish()
//     }
// }

// impl<K, V> Clone for KeyIndexMap<K, V>
// where
//     K: Clone,
//     V: Clone,
// {
//     /// Rcに関してはそのままRcとしてクローンされる
//     /// 問題あればDeep Cloneを実装すること
//     fn clone(&self) -> Self {
//         KeyIndexMap {
//             entries: self.entries.clone(),
//             table: self.table.clone(),
//         }
//     }
// }



/// IndexMap
/// 連続領域を保証するHashMap
/// 
/// # Safety
/// table, hashes は table_* メソッドが責任をもつこと 更新とか
#[derive(Clone, Debug)]
pub struct IndexMap<K, V, S = ahash::RandomState> {
    pub values: Vec<V>,
    pub keys: Vec<K>,
    pub hashes: Vec<u64>,
    pub table: HashTable<usize>,
    pub hash_builder: S,
}

impl<K, V, S> IndexMap<K, V, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        IndexMap {
            values: Vec::new(),
            keys: Vec::new(),
            hashes: Vec::new(),
            table: HashTable::new(),
            hash_builder,
        }
    }

    pub fn new() -> Self
    where
        S: Default,
    {
        IndexMap {
            values: Vec::new(),
            keys: Vec::new(),
            hashes: Vec::new(),
            table: HashTable::new(),
            hash_builder: S::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter_values(&self) -> std::slice::Iter<'_, V> {
        self.values.iter()
    }

    pub fn iter_keys(&self) -> std::slice::Iter<'_, K> {
        self.keys.iter()
    }

    pub fn iter_key_value(&self) -> impl Iterator<Item = (&K, &V)> {
        self.keys.iter().zip(self.values.iter())
    }

    pub fn values(&self) -> &Vec<V> {
        &self.values
    }

    pub fn keys(&self) -> &Vec<K> {
        &self.keys
    }

    /// hash util
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// override
    /// 完全な整合性が必要
    /// keyに対するidxを更新し、更新したidxを返す
    /// 存在しない場合はNone
    unsafe fn table_override(&mut self, key: &K, idx: &usize) -> Option<usize> {
        let hash = self.hash_key(key);
        match self.table.find_entry(hash, |&i| self.keys[i] == *key) {
            Ok(mut occ) => {
                // idxの上書きだけ
                *occ.get_mut() = *idx;
                Some(*idx)
            }
            Err(_) => {
                None
            }
        }
    }

    /// append
    /// 完全な整合性が必要
    /// hashesとtableを更新する
    unsafe fn table_append(&mut self, key: &K, idx: &usize) {
        let hash = self.hash_key(key);
        self.hashes.push(hash);
        self.table.insert_unique(
            hash,
            *idx,
            |&i| self.hashes[i]
        );
    }

    /// get
    /// とくに注意なし 不可変参照なので
    fn table_get(&self, key: &K) -> Option<usize> {
        let hash = self.hash_key(key);
        self.table.find(
            hash, 
            |&i| self.keys[i] == *key
        ).copied()
    }

    /// contains
    /// とくに注意なし 不可変参照なので
    fn table_contains(&self, key: &K) -> bool {
        let hash = self.hash_key(key);
        self.table.find(
            hash, 
            |&i| self.keys[i] == *key
        ).is_some()
    }

    /// remove
    /// 完全な整合性が必要
    /// hashesはswap_removeされます
    unsafe fn table_swap_remove(&mut self, key: &K) -> Option<usize> {
        let hash = self.hash_key(key);
        if let Ok(entry) = self.table.find_entry(
            hash,
            |&i| self.keys[i] == *key
        ) {
            let (odl_idx, _) = entry.remove();
            self.hashes.swap_remove(odl_idx);
            Some(odl_idx)
        } else {
            None
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(idx) = self.table_get(key) {
            unsafe {
                Some(self.values.get_unchecked(idx))
            }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(idx) = self.table_get(key) {
            unsafe {
                Some(self.values.get_unchecked_mut(idx))
            }
        } else {
            None
        }
    }

    pub fn get_index(&self, index: usize) -> Option<&V> {
        self.values.get(index)
    }

    pub fn get_index_mut(&mut self, index: usize) -> Option<&mut V> {
        self.values.get_mut(index)
    }

    pub fn get_key_index(&self, index: usize) -> Option<&K> {
        self.keys.get(index)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.table_get(key).is_some()
    }

    pub fn insert(&mut self, key: &K, value: V) -> Option<InsertResult<K, V>> {
        if let Some(idx) = self.table_get(key) {
            // K が Rc の場合を考慮して すべて差し替える
            unsafe {
                self.table_override(key, &idx);
            }
            let old_value = Some(std::mem::replace(&mut self.values[idx], value));
            let old_key = Some(std::mem::replace(&mut self.keys[idx], key.clone()));
            Some(InsertResult {
                old_value: old_value.unwrap(),
                old_key:  old_key.unwrap(),
            })
        } else {
            // New key, insert entry
            let idx = self.values.len();
            unsafe {
                self.table_append(key, &idx);
            }
            self.keys.push(key.clone());
            self.values.push(value);
            None
        }
    }

    pub fn entry_mut<'a>(&'a mut self, key: &'a K) -> EntryMut<'a, K, V, S> {
        if let Some(idx) = self.table_get(key) {
            unsafe {
                EntryMut::Occupied {
                    key: self.keys.get_unchecked(idx),
                    value: self.values.get_unchecked_mut(idx),
                    index: idx,
                }
            }
        } else {
            EntryMut::Vacant { key , map: self }
        }
    }

    pub fn swap_remove(&mut self, key: &K) -> Option<V> {
        if let Some(idx) = self.table_get(key) {
            let last_idx = self.values.len() - 1;
            if idx == last_idx {
                // 最後の要素を削除する場合
                unsafe {
                    self.table_swap_remove(key);
                }
                self.keys.pop();
                return Some(self.values.pop().unwrap());
            } else {
                let last_idx_key = self.keys[last_idx].clone();
                unsafe {
                    // keyとの整合性があるうちに削除予定のをtableから消す ここでhashesがswap_removeされる
                    // last_idxの要素がswapで移動してくる
                    self.table_swap_remove(key);
                    // 移動させられた要素のtableを再登録
                    // 登録されていた前のidxに対するkeyはまだ整合性が取れているので問題ない
                    self.table_override(&last_idx_key, &idx);
                }
                // swap_remove ここで実際にtableのidxとvalues, keys, hashesの整合性が回復
                let value = self.values.swap_remove(idx);
                self.keys.swap_remove(idx);
                Some(value)
            }
        } else {
            None
        }
    }
}

pub enum EntryMut<'a, K, V, S> {
    Occupied { key: &'a K, value: &'a mut V, index: usize },
    Vacant { key: &'a K , map: &'a mut IndexMap<K, V, S> },
}

impl<'a, K, V, S> EntryMut<'a, K, V, S>
where 
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    pub fn is_occupied(&self) -> bool {
        matches!(self, EntryMut::Occupied { .. })
    }

    pub fn or_insert_with<F>(self, value: F) -> &'a mut V
    where
        F: FnOnce() -> V,
        K: Clone,
    {
        match self {
            EntryMut::Occupied { value: v, .. } => v,
            EntryMut::Vacant { key, map } => {
                map.insert(key, value());
                map.get_mut(key).unwrap()
            }
        }
    }
}

pub struct InsertResult<K, V> {
    pub old_value: V,
    pub old_key: K,
}

// pub struct IndexMapVec<T> {
//     pub vec: IndexMapRawVec<T>,
//     pub len: usize,
// }

// impl<T> IndexMapVec<T> {
//     pub fn new() -> Self {
//         IndexMapVec {
//             vec: IndexMapRawVec::with_capacity(0),
//             len: 0,
//         }
//     }

//     pub fn with_capacity(cap: usize) -> Self {
//         IndexMapVec {
//             vec: IndexMapRawVec::with_capacity(cap),
//             len: 0,
//         }
//     }

//     /// pushする
//     /// reallocされた場合、メモリリアドレスが変わるので、その差分をisizeで返す
//     pub fn push(&mut self, value: T) -> i64 {
//         let front_ptr = self.vec.val_ptr();
//         if self.len == self.vec.cap {
//             self.vec.reallocate(self.vec.cap * 2);
//         }
//         unsafe {
//             *self.vec.val_ptr().add(self.len) = value;
//         }
//         self.len += 1;
//         let new_ptr = self.vec.val_ptr();
//         (new_ptr as i64) - (front_ptr as i64)
//     }

//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn pop(&mut self) -> Option<T> {
//         if self.len == 0 {
//             None
//         } else {
//             self.len -= 1;
//             unsafe {
//                 Some(std::ptr::read(self.vec.val_ptr().add(self.len)))
//             }
//         }
//     }

//     pub fn get(&self, index: usize) -> Option<&T> {
//         if index >= self.len {
//             None
//         } else {
//             unsafe {
//                 Some(&*self.vec.val_ptr().add(index))
//             }
//         }
//     }

//     pub unsafe fn get_unchecked(&self, index: usize) -> &T {
//         unsafe {
//             &*self.vec.val_ptr().add(index)
//         }
//     }

//     pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
//         unsafe {
//             &mut *self.vec.val_ptr().add(index)
//         }
//     }

//     pub fn iter(&self) -> std::slice::Iter<'_, T> {
//         unsafe {
//             std::slice::from_raw_parts(self.vec.val_ptr(), self.len).iter()
//         }
//     }

//     pub fn as_slice(&self) -> &[T] {
//         unsafe {
//             std::slice::from_raw_parts(self.vec.val_ptr(), self.len)
//         }
//     }
// }

// impl<T> Clone for IndexMapVec<T> 
// where T: Clone
// {
//     fn clone(&self) -> Self {
//         let new_vec = IndexMapRawVec::<T>::with_capacity(self.vec.cap);
//         for i in 0..self.len {
//             unsafe {
//                 let src = self.vec.val_ptr().add(i);
//                 let dst = new_vec.val_ptr().add(i);
//                 dst.write((*src).clone());
//             }
//         }
//         IndexMapVec {
//             vec: new_vec,
//             len: self.len,
//         }
//     }
// }

// impl<T> Debug for IndexMapVec<T> 
// where T: Debug
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let slice = unsafe {
//             std::slice::from_raw_parts(self.vec.val_ptr(), self.len)
//         };
//         f.debug_list().entries(slice.iter()).finish()
//     }
// }

// pub struct IndexMapRawVec<T> {
//     pub ptr: NonNull<T>,
//     pub cap: usize,
// }

// impl<T> IndexMapRawVec<T> {
//     pub fn with_capacity(cap: usize) -> Self {
//         let layout = std::alloc::Layout::array::<T>(cap).unwrap();
//         let ptr = unsafe { std::alloc::alloc(layout) } as *mut T;
//         IndexMapRawVec {
//             ptr: NonNull::new(ptr).unwrap(),
//             cap,
//         }
//     }

//     pub fn reallocate(&mut self, new_cap: usize) {
//         let old_layout = std::alloc::Layout::array::<T>(self.cap).unwrap();
//         let new_layout = std::alloc::Layout::array::<T>(new_cap).unwrap();
//         let new_ptr = unsafe { std::alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size()) } as *mut T;
//         self.ptr = NonNull::new(new_ptr).unwrap();
//         self.cap = new_cap;
//     }

//     pub fn val_ptr(&self) -> *mut T {
//         self.ptr.as_ptr()
//     }
// }

// impl<T> Drop for IndexMapRawVec<T> {
//     fn drop(&mut self) {
//         let layout = std::alloc::Layout::array::<T>(self.cap).unwrap();
//         unsafe {
//             std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
//         }
//     }   
// }