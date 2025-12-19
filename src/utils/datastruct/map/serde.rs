use std::{
    fmt,
    hash::Hash,
    marker::PhantomData,
};

use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize};

use super::{Entry, KeyIndexMap, KeyRc};

impl<K, V> Serialize for KeyIndexMap<K, V>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for entry in &self.entries {
            map.serialize_entry(entry.key.rc.as_ref(), &entry.value)?;
        }
        map.end()
    }
}

impl<'de, K, V> Deserialize<'de> for KeyIndexMap<K, V>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct KeyIndexMapVisitor<K, V>(PhantomData<(K, V)>);

        impl<'de, K, V> serde::de::Visitor<'de> for KeyIndexMapVisitor<K, V>
        where
            K: Deserialize<'de> + Hash + Eq,
            V: Deserialize<'de>,
        {
            type Value = KeyIndexMap<K, V>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map of {key: value} for KeyIndexMap")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let capacity = access.size_hint().unwrap_or(0);
                let mut entries: Vec<Entry<KeyRc<K>, V>> = Vec::with_capacity(capacity);
                let mut table: HashMap<KeyRc<K>, usize, ahash::RandomState> =
                    HashMap::with_capacity_and_hasher(capacity, ahash::RandomState::new());

                while let Some((key, value)) = access.next_entry::<K, V>()? {
                    if let Some((_existing_key, &idx)) = table.get_key_value(&key) {
                        entries[idx].value = value;
                        continue;
                    }

                    let idx = entries.len();
                    let key_rc = KeyRc::new(key);
                    entries.push(Entry {
                        key: key_rc.clone(),
                        value,
                    });
                    table.insert(key_rc, idx);
                }

                Ok(KeyIndexMap { entries, table })
            }
        }

        deserializer.deserialize_map(KeyIndexMapVisitor(PhantomData))
    }
}