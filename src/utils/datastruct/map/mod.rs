use hashbrown::HashTable;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hasher;

pub mod serde;

/// IndexMap
/// 連続領域を保証するHashMap
/// 
/// # Safety
/// table, hashes は table_* メソッドが責任をもつこと 更新とか
/// 
/// いじんないじんないじんな いじったならあらゆるUnitTest書いて通せ
#[derive(Clone, Debug)]
pub struct IndexMap<K, V, S = ahash::RandomState> {
    values: Vec<V>,
    index_set: IndexSet<K, S>,
}

impl<K, V, S> IndexMap<K, V, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        IndexMap {
            values: Vec::new(),
            index_set: IndexSet::with_hasher(hash_builder),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self
    where
        S: Default,
    {
        IndexMap {
            values: Vec::with_capacity(capacity),
            index_set: IndexSet::with_capacity(capacity),
        }
    }

    pub fn new() -> Self
    where
        S: Default,
    {
        IndexMap {
            values: Vec::new(),
            index_set: IndexSet::new(),
        }
    }

    pub fn with_hasher_capacity(hash_builder: S, capacity: usize) -> Self {
        IndexMap {
            values: Vec::with_capacity(capacity),
            index_set: IndexSet::with_hasher_capacity(hash_builder, capacity),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn iter_values(&self) -> std::slice::Iter<'_, V> {
        self.values.iter()
    }

    #[inline]
    pub fn iter_keys(&self) -> std::slice::Iter<'_, K> {
        self.index_set.keys.iter()
    }

    #[inline]
    pub fn iter(&self) -> IndexMapIter<'_, K, V, S> {
        IndexMapIter {
            map: self,
            index: 0,
        }
    }

    #[inline]
    pub fn values(&self) -> &Vec<V> {
        &self.values
    }

    #[inline]
    pub fn keys(&self) -> &Vec<K> {
        &self.index_set.keys()
    }

    #[inline]
    pub fn as_index_set(&self) -> &IndexSet<K, S> {
        &self.index_set
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        if let Some(idx) = self.index_set.table_get(key) {
            unsafe {
                Some(self.values.get_unchecked(idx))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        if let Some(idx) = self.index_set.table_get(key) {
            unsafe {
                Some(self.values.get_unchecked_mut(idx))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn get_with_index(&self, index: usize) -> Option<&V> {
        self.values.get(index)
    }

    #[inline]
    pub fn get_with_index_mut(&mut self, index: usize) -> Option<&mut V> {
        self.values.get_mut(index)
    }

    #[inline]
    pub fn get_key_with_index(&self, index: usize) -> Option<&K> {
        self.index_set.get_with_index(index)
    }

    #[inline]
    pub fn get_key_value_with_index(&self, index: usize) -> Option<(&K, &V)> {
        if index < self.len() {
            unsafe {
                Some((
                    self.index_set.keys.get_unchecked(index),
                    self.values.get_unchecked(index),
                ))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn get_index<Q: ?Sized>(&self, key: &Q) -> Option<usize> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        self.index_set.table_get(key)
    }

    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        self.index_set.contains_key(key)
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<InsertResult<K, V>> {
        if let Some(idx) = self.index_set.table_get(&key) {
            // K が Rc の場合を考慮して すべて差し替える
            unsafe {
                self.index_set.table_override(&key, &idx);
            }
            let old_value = Some(std::mem::replace(&mut self.values[idx], value));
            let old_key = Some(std::mem::replace(&mut self.index_set.keys[idx], key.clone()));
            Some(InsertResult {
                old_value: old_value.unwrap(),
                old_key:  old_key.unwrap(),
            })
        } else {
            // New key, insert entry
            let idx = self.values.len();
            unsafe {
                self.index_set.table_append(&key, &idx);
            }
            self.index_set.keys.push(key.clone());
            self.values.push(value);
            None
        }
    }

    #[inline]
    pub fn entry_mut<'a>(&'a mut self, key: K) -> EntryMut<'a, K, V, S> {
        if let Some(idx) = self.index_set.table_get(&key) {
            unsafe {
                EntryMut::Occupied {
                    key: key,
                    value: self.values.get_unchecked_mut(idx),
                    index: idx,
                }
            }
        } else {
            EntryMut::Vacant { key , map: self }
        }
    }

    #[inline]
    pub fn swap_remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        if let Some(idx) = self.index_set.table_get(key) {
            let last_idx = self.values.len() - 1;
            if idx == last_idx {
                // 最後の要素を削除する場合
                unsafe {
                    self.index_set.table_swap_remove(key);
                }
                self.index_set.keys.pop();
                return Some(self.values.pop().unwrap());
            } else {
                let last_idx_key = self.index_set.keys[last_idx].clone();
                unsafe {
                    // keyとの整合性があるうちに削除予定のをtableから消す ここでhashesがswap_removeされる
                    // last_idxの要素がswapで移動してくる
                    self.index_set.table_swap_remove(key);
                    // 移動させられた要素のtableを再登録
                    // 登録されていた前のidxに対するkeyはまだ整合性が取れているので問題ない
                    self.index_set.table_override(last_idx_key.borrow(), &idx);
                }
                // swap_remove ここで実際にtableのidxとvalues, keys, hashesの整合性が回復
                let value = self.values.swap_remove(idx);
                self.index_set.keys.swap_remove(idx);
                Some(value)
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn from_kv_vec(k_vec: Vec<K>, v_vec: Vec<V>) -> Self
    where
        S: std::hash::BuildHasher + Default,
    {
        let hash_builder = S::default();
        let mut map = IndexMap::with_hasher(hash_builder);
        for (k, v) in k_vec.into_iter().zip(v_vec.into_iter()) {
            let idx = map.values.len();
            unsafe {
                map.index_set.table_append(&k, &idx);
            }
            map.index_set.keys.push(k);
            map.values.push(v);
        }
        map
    }
}

pub struct IndexMapIter<'a, K, V, S> {
    pub map: &'a IndexMap<K, V, S>,
    pub index: usize,
}

impl <'a, K, V, S> Iterator for IndexMapIter<'a, K, V, S> 
where 
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.map.len() {
            unsafe {
                let k = self.map.index_set.keys.get_unchecked(self.index);
                let v = self.map.values.get_unchecked(self.index);
                self.index += 1;
                Some((k, v))
            }
        } else {
            None
        }
    } 
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.map.len() - self.index;
        (remaining, Some(remaining))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n;
        self.next()
    }
}

pub enum EntryMut<'a, K, V, S> {
    Occupied { key: K, value: &'a mut V, index: usize },
    Vacant { key: K , map: &'a mut IndexMap<K, V, S> },
}

impl<'a, K, V, S> EntryMut<'a, K, V, S>
where 
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    #[inline]
    pub fn is_occupied(&self) -> bool {
        matches!(self, EntryMut::Occupied { .. })
    }

    #[inline]
    pub fn or_insert_with<F>(self, value: F) -> &'a mut V
    where
        F: FnOnce() -> V,
        K: Clone,
    {
        match self {
            EntryMut::Occupied { value: v, .. } => v,
            EntryMut::Vacant { key, map } => {
                map.insert(key.clone(), value());
                map.get_mut(&key).unwrap()
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct InsertResult<K, V> {
    pub old_value: V,
    pub old_key: K,
}

#[derive(Clone, Debug)]
pub struct IndexSet<K, S = ahash::RandomState> {
    keys: Vec<K>,
    hashes: Vec<u64>,
    table: HashTable<usize>,
    hash_builder: S,
}

impl<K, S> IndexSet<K, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: std::hash::BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        IndexSet {
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
        IndexSet {
            keys: Vec::new(),
            hashes: Vec::new(),
            table: HashTable::new(),
            hash_builder: S::default(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self
    where
        S: Default,
    {
        IndexSet {
            keys: Vec::with_capacity(capacity),
            hashes: Vec::with_capacity(capacity),
            table: HashTable::with_capacity(capacity),
            hash_builder: S::default(),
        }
    }

    pub fn with_hasher_capacity(hash_builder: S, capacity: usize) -> Self {
        IndexSet {
            keys: Vec::with_capacity(capacity),
            hashes: Vec::with_capacity(capacity),
            table: HashTable::with_capacity(capacity),
            hash_builder,
        }
    }

    /// hash util
    #[inline]
    fn hash_key<Q: ?Sized>(&self, key: &Q) -> u64 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// override
    /// 完全な整合性が必要
    /// keyに対するidxを更新し、更新したidxを返す
    /// 存在しない場合はNone
    #[inline]
    unsafe fn table_override<Q: ?Sized>(&mut self, key: &Q, idx: &usize) -> Option<usize> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        let hash = self.hash_key(key);
        match self.table.find_entry(hash, |&i| self.keys[i].borrow() == key) {
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
    #[inline]
    unsafe fn table_append<Q: ?Sized>(&mut self, key: &Q, idx: &usize) 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
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
    #[inline]
    fn table_get<Q: ?Sized>(&self, key: &Q) -> Option<usize> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        let hash = self.hash_key(key);
        self.table.find(
            hash, 
            |&i| self.keys[i].borrow() == key
        ).copied()
    }

    /// remove
    /// 完全な整合性が必要
    /// hashesはswap_removeされます
    #[inline]
    unsafe fn table_swap_remove<Q: ?Sized>(&mut self, key: &Q) -> Option<usize> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        let hash = self.hash_key(key);
        if let Ok(entry) = self.table.find_entry(
            hash,
            |&i| self.keys[i].borrow() == key
        ) {
            let (odl_idx, _) = entry.remove();
            self.hashes.swap_remove(odl_idx);
            Some(odl_idx)
        } else {
            None
        }
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        self.table_get(key).is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn get_index<Q: ?Sized>(&self, key: &Q) -> Option<usize> 
    where 
        K: Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        self.table_get(key)
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, K> {
        self.keys.iter()
    }

    #[inline]
    pub fn keys(&self) -> &Vec<K> {
        &self.keys
    }

    #[inline]
    pub fn get_with_index(&self, index: usize) -> Option<&K> {
        self.keys.get(index)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // まずは比較しやすい型でテスト
    type M = IndexMap<u64, i64>;

    fn assert_internal_invariants(map: &M) {
        // 長さが揃っていること
        assert_eq!(map.values.len(), map.index_set.keys.len(), "values/keys len mismatch");
        assert_eq!(map.values.len(), map.index_set.hashes.len(), "values/hashes len mismatch");

        // table_get が返す idx が範囲内で、keys/values と一致すること
        for (i, k) in map.index_set.keys.iter().enumerate() {
            let idx = map.index_set.table_get(k).expect("table_get must find existing key");
            assert_eq!(idx, i, "table idx mismatch for key");
        }

        // 逆方向も確認
        // 重複キー禁止 + contains/get の整合
        for i in 0..map.len() {
            let k = &map.index_set.keys[i];
            assert!(map.contains_key(k), "contains_key false for existing key");
            let v = map.get(k).expect("get must return for existing key");
            assert_eq!(*v, map.values[i], "get value mismatch");
        }

        // キー重複が無いこと
        // O(n^2) だけどユニットテストならOK
        for i in 0..map.index_set.keys.len() {
            for j in (i + 1)..map.index_set.keys.len() {
                assert!(map.index_set.keys[i] != map.index_set.keys[j], "duplicate keys detected");
            }
        }
    }

    fn assert_equals_oracle(map: &M, oracle: &HashMap<u64, i64>) {
        assert_eq!(map.len(), oracle.len(), "len mismatch");

        // 全キーが一致し、値も一致すること
        for (k, v) in oracle.iter() {
            let got = map.get(k).copied();
            assert_eq!(got, Some(*v), "value mismatch for key={k}");
        }

        // map 側に余計なキーがないこと（oracle で確認）
        for (k, v) in map.iter() {
            assert_eq!(oracle.get(k).copied(), Some(*v), "extra/mismatch entry key={k}");
        }
    }

    #[test]
    fn basic_insert_get_overwrite() {
        let mut m = M::new();

        assert_eq!(m.insert(1, 10), None);
        assert_eq!(m.insert(2, 20), None);
        assert_eq!(m.get(&1).copied(), Some(10));
        assert_eq!(m.get(&2).copied(), Some(20));

        // overwrite
        let old = m.insert(1, 99).unwrap();
        assert_eq!(old.old_key, 1);
        assert_eq!(old.old_value, 10);
        assert_eq!(m.get(&1).copied(), Some(99));

        assert_internal_invariants(&m);
    }

    #[test]
    fn swap_remove_last_and_middle() {
        let mut m = M::new();
        for i in 0..10 {
            m.insert(i, (i as i64) * 10);
        }

        // last remove
        let v = m.swap_remove(&9);
        assert_eq!(v, Some(90));
        assert!(m.get(&9).is_none());

        // middle remove
        let v = m.swap_remove(&3);
        assert_eq!(v, Some(30));
        assert!(m.get(&3).is_none());

        assert_internal_invariants(&m);
    }

    #[test]
    fn entry_or_insert_with_works() {
        let mut m = M::new();

        let v = m.entry_mut(7).or_insert_with(|| 123);
        assert_eq!(*v, 123);

        // 2回目は既存参照が返る
        let v2 = m.entry_mut(7).or_insert_with(|| 999);
        assert_eq!(*v2, 123);

        assert_internal_invariants(&m);
    }

    #[test]
    fn compare_with_std_hashmap_small_scripted() {
        let mut m = M::new();
        let mut o = HashMap::<u64, i64>::new();

        // 混ぜた操作を固定シナリオで
        for i in 0..50u64 {
            m.insert(i, i as i64);
            o.insert(i, i as i64);
        }

        for i in 0..50u64 {
            if i % 3 == 0 {
                let a = m.swap_remove(&i);
                let b = o.remove(&i);
                assert_eq!(a, b);
            }
        }

        for i in 0..50u64 {
            if i % 5 == 0 {
                m.insert(i, (i as i64) * 100);
                o.insert(i, (i as i64) * 100);
            }
        }

        assert_internal_invariants(&m);
        assert_equals_oracle(&m, &o);
    }

    #[test]
    fn randomized_ops_compare_with_oracle() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut m = M::new();
        let mut o = HashMap::<u64, i64>::new();

        // ある程度衝突や削除を踏む
        const STEPS: usize = 30_000;
        const KEY_SPACE: u64 = 2_000;

        for _ in 0..STEPS {
            let op = rng.gen_range(0..100);
            let k = rng.gen_range(0..KEY_SPACE);
            match op {
                // insert (多め)
                0..=59 => {
                    let v = rng.gen_range(-1_000_000..=1_000_000);
                    let a = m.insert(k, v);
                    let b = o.insert(k, v);

                    match (a, b) {
                        (None, None) => {}
                        (Some(ir), Some(old)) => {
                            assert_eq!(ir.old_key, k);
                            assert_eq!(ir.old_value, old);
                        }
                        _ => panic!("insert mismatch"),
                    }
                }
                // swap_remove
                60..=79 => {
                    let a = m.swap_remove(&k);
                    let b = o.remove(&k);
                    assert_eq!(a, b);
                }
                // get
                80..=94 => {
                    let a = m.get(&k).copied();
                    let b = o.get(&k).copied();
                    assert_eq!(a, b);
                }
                // contains
                _ => {
                    let a = m.contains_key(&k);
                    let b = o.contains_key(&k);
                    assert_eq!(a, b);
                }
            }

            // たまに内部整合をチェック（重いので間引く）
            if rng.gen_ratio(1, 200) {
                assert_internal_invariants(&m);
                assert_equals_oracle(&m, &o);
            }
        }

        // 最後に必ず一致
        assert_internal_invariants(&m);
        assert_equals_oracle(&m, &o);
    }

    #[test]
    fn empty_map_basics() {
        let m = M::new();

        assert_eq!(m.len(), 0);
        assert!(m.get(&123).is_none());
        assert!(!m.contains_key(&123));
        // 空でも長さ整合は成立
        assert_eq!(m.values.len(), 0);
        assert_eq!(m.index_set.keys.len(), 0);
        assert_eq!(m.index_set.hashes.len(), 0);
    }

    #[test]
    fn swap_remove_single_element_roundtrip() {
        let mut m = M::new();
        m.insert(42, -7);
        assert_internal_invariants(&m);

        let v = m.swap_remove(&42);
        assert_eq!(v, Some(-7));
        assert_eq!(m.len(), 0);
        assert!(m.get(&42).is_none());
        assert!(!m.contains_key(&42));

        assert_internal_invariants(&m);
    }

    #[test]
    fn remove_then_reinsert_same_key() {
        let mut m = M::new();

        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        assert_internal_invariants(&m);

        assert_eq!(m.swap_remove(&2), Some(20));
        assert!(m.get(&2).is_none());
        assert_internal_invariants(&m);

        // 同じキーを再挿入しても table が壊れないこと
        assert_eq!(m.insert(2, 200), None);
        assert_eq!(m.get(&2).copied(), Some(200));
        assert_internal_invariants(&m);
    }

    #[test]
    fn from_kv_vec_builds_valid_map() {
        let keys = vec![1u64, 2u64, 3u64, 10u64];
        let values = vec![10i64, 20i64, 30i64, 100i64];

        let m = M::from_kv_vec(keys.clone(), values.clone());
        assert_eq!(m.len(), 4);

        // 順序と内容が一致
        assert_eq!(m.index_set.keys, keys);
        assert_eq!(m.values, values);

        assert_internal_invariants(&m);
    }

    #[test]
    fn iter_order_matches_internal_storage_even_after_removes() {
        let mut m = M::new();
        for i in 0..8u64 {
            m.insert(i, (i as i64) + 100);
        }
        assert_internal_invariants(&m);

        // いくつか消して、内部順序が変わっても iter が keys/values と整合すること
        assert_eq!(m.swap_remove(&0), Some(100));
        assert_eq!(m.swap_remove(&5), Some(105));
        assert_internal_invariants(&m);

        let collected: Vec<(u64, i64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        let expected: Vec<(u64, i64)> = m.index_set.keys.iter().copied().zip(m.values.iter().copied()).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn num_23257_issue() {
        const ITER_NUM: u64 = 223259;
        let mut map: IndexMap<u64, Vec<u64>> = IndexMap::new();
        for i in 0..ITER_NUM {
            map.entry_mut(0).or_insert_with(Vec::new).push(i);
        }
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&0).unwrap().len() as u64, ITER_NUM);
    }
}
