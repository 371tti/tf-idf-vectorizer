use std::{
    fmt,
    hash::Hash,
    marker::PhantomData,
};

use serde::{Deserialize, Deserializer, Serialize, ser::SerializeStruct};

use crate::utils::datastruct::map::IndexMap;

impl<K, V> Serialize for IndexMap<K, V>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("IndexMap", 2)?;
        state.serialize_field("values", &self.values)?;
        state.serialize_field("keys", &self.index_set.keys)?;
        state.end()
    }
}

impl<'de, K, V> Deserialize<'de> for IndexMap<K, V>
where
    K: Deserialize<'de> + Hash + Eq + Clone,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Debug)]
        enum Field {
            Values,
            Keys,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`values` or `keys`")
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        match v {
                            "values" => Ok(Field::Values),
                            "keys" => Ok(Field::Keys),
                            _ => Err(E::unknown_field(v, FIELDS)),
                        }
                    }

                    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        match v {
                            b"values" => Ok(Field::Values),
                            b"keys" => Ok(Field::Keys),
                            _ => {
                                let s = std::str::from_utf8(v).unwrap_or("");
                                Err(E::unknown_field(s, FIELDS))
                            }
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct IndexMapVisitor<K, V>(PhantomData<(K, V)>);

        const FIELDS: &[&str] = &["values", "keys"];

        impl<'de, K, V> serde::de::Visitor<'de> for IndexMapVisitor<K, V>
        where
            K: Deserialize<'de> + Hash + Eq + Clone,
            V: Deserialize<'de>,
        {
            type Value = IndexMap<K, V>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an IndexMap serialized as { values: Vec<V>, keys: Vec<K> }")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                use serde::de::Error as DeError;

                let mut values: Option<Vec<V>> = None;
                let mut keys: Option<Vec<K>> = None;

                while let Some(field) = access.next_key::<Field>()? {
                    match field {
                        Field::Values => {
                            if values.is_some() {
                                return Err(DeError::duplicate_field("values"));
                            }
                            values = Some(access.next_value()?);
                        }
                        Field::Keys => {
                            if keys.is_some() {
                                return Err(DeError::duplicate_field("keys"));
                            }
                            keys = Some(access.next_value()?);
                        }
                    }
                }

                let values = values.ok_or_else(|| DeError::missing_field("values"))?;
                let keys = keys.ok_or_else(|| DeError::missing_field("keys"))?;

                if keys.len() != values.len() {
                    return Err(DeError::custom("IndexMap deserialize error: keys and values length mismatch"));
                }

                Ok(IndexMap::from_kv_vec(keys, values))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                use serde::de::Error as DeError;

                let values: Vec<V> = seq
                    .next_element()?
                    .ok_or_else(|| DeError::invalid_length(0, &self))?;
                let keys: Vec<K> = seq
                    .next_element()?
                    .ok_or_else(|| DeError::invalid_length(1, &self))?;

                if keys.len() != values.len() {
                    return Err(DeError::custom("IndexMap deserialize error: keys and values length mismatch"));
                }

                Ok(IndexMap::from_kv_vec(keys, values))
            }
        }

        deserializer.deserialize_struct("IndexMap", FIELDS, IndexMapVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_json_map_format_preserves_order_and_lookup() {
        let mut m = IndexMap::<String, i64>::new();
        m.insert("a".to_string(), 10);
        m.insert("b".to_string(), 20);
        m.insert("c".to_string(), 30);

        let s = serde_json::to_string(&m).unwrap();
        let de: IndexMap<String, i64> = serde_json::from_str(&s).unwrap();

        assert_eq!(de.index_set.keys, m.index_set.keys);
        assert_eq!(de.values, m.values);
        assert_eq!(de.len(), 3);
        assert_eq!(de.index_set.hashes.len(), de.len());

        for (k, v) in m.iter() {
            assert_eq!(de.get(k).copied(), Some(*v));
        }
    }

    #[test]
    fn serde_roundtrip_bincode_seq_format_works() {
        let mut m = IndexMap::<u64, i64>::new();
        for i in 0..100u64 {
            m.insert(i, (i as i64) * -7);
        }

        let bytes = bincode::serialize(&m).unwrap();
        let de: IndexMap<u64, i64> = bincode::deserialize(&bytes).unwrap();

        assert_eq!(de.index_set.keys, m.index_set.keys);
        assert_eq!(de.values, m.values);
        assert_eq!(de.index_set.hashes.len(), de.len());

        for (k, v) in m.iter() {
            assert_eq!(de.get(k).copied(), Some(*v));
        }
    }

    #[test]
    fn serde_rejects_len_mismatch_json() {
        // values.len != keys.len
        let bad = r#"{\"values\":[1,2,3],\"keys\":[\"a\",\"b\"]}"#;
        let res = serde_json::from_str::<IndexMap<String, i32>>(bad);
        assert!(res.is_err());
    }

    #[test]
    fn serde_rejects_duplicate_fields_json() {
        // JSON的には非推奨だが、パーサが許すケースがあるので、ここで弾けると安心
        let dup = r#"{\"values\":[1],\"values\":[2],\"keys\":[\"a\"]}"#;
        let res = serde_json::from_str::<IndexMap<String, i32>>(dup);
        assert!(res.is_err());
    }
}