use num::Num;
use serde::{Serialize, Serializer, Deserialize, Deserializer};
use serde::ser::SerializeStruct;
use crate::utils::math::vector::ZeroSpVecTrait;

use super::ZeroSpVec;

impl<N> Serialize for ZeroSpVec<N>
where
    N: Num + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        // シリアライズするフィールドは len, nnz, entries とする
        let mut state = serializer.serialize_struct("ZeroSpVec", 3)?;
        state.serialize_field("len", &(self.len as u64))?;
        state.serialize_field("nnz", &(self.nnz as u64))?;
        
        // entries: (index, value) のVecとして順序付きに出力する
        let mut entries = Vec::with_capacity(self.nnz);
        //  entries = self.raw_iter().map(|(idx, entry)| (idx as u64, *entry)).collect();
        unsafe {
            for i in 0..self.nnz {
                let idx = *self.ind_ptr().add(i);
                let val = *self.val_ptr().add(i);
                entries.push((idx as u64, val));
            }
        }
        state.serialize_field("entries", &entries)?;
        state.end()
    }
}

impl<'de, N> Deserialize<'de> for ZeroSpVec<N>
where
    N: Num + Deserialize<'de> + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        use serde::de::{Visitor, MapAccess, Error as DeError};
        use std::fmt;

        struct ZeroSpVecVisitor<N> {
            marker: std::marker::PhantomData<N>,
        }

        impl<'de, N> Visitor<'de> for ZeroSpVecVisitor<N>
        where N: Num + Deserialize<'de> + Copy {
            type Value = ZeroSpVec<N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "struct ZeroSpVec")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where A: MapAccess<'de> {
                let mut len = None;
                let mut nnz = None;
                let mut entries = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "len" => len = Some(map.next_value::<u64>()? as usize),
                        "nnz" => nnz = Some(map.next_value::<u64>()? as usize),
                        "entries" => entries = Some(map.next_value::<Vec<(u64, N)>>()?),
                        _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
                    }
                }
                let len = len.ok_or_else(|| DeError::missing_field("len"))?;
                let nnz = nnz.ok_or_else(|| DeError::missing_field("nnz"))?;
                let entries = entries.ok_or_else(|| DeError::missing_field("entries"))?;
                let mut vec = ZeroSpVec::with_capacity(nnz);
                vec.len = len;
                for (index, value) in entries {
                    unsafe { vec.raw_push(index as usize, value) };
                }
                Ok(vec)
            }
        }
        deserializer.deserialize_struct(
            "ZeroSpVec",
            &["len", "nnz", "entries"],
            ZeroSpVecVisitor { marker: std::marker::PhantomData },
        )
    }
}