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
        
        // inds/vals: バッファをそのまま配列として出力する（タプルを避ける）
        let mut inds: Vec<u32> = Vec::with_capacity(self.nnz);
        let mut vals: Vec<N> = Vec::with_capacity(self.nnz);
        unsafe {
            for i in 0..self.nnz {
                inds.push(*self.ind_ptr().add(i));
                vals.push(*self.val_ptr().add(i));
            }
        }
        state.serialize_field("inds", &inds)?;
        state.serialize_field("vals", &vals)?;
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
                let mut entries: Option<Vec<(u64, N)>> = None; // 後方互換
                let mut inds: Option<Vec<u32>> = None;
                let mut vals: Option<Vec<N>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "len" => len = Some(map.next_value::<u64>()? as usize),
                        "nnz" => nnz = Some(map.next_value::<u64>()? as usize),
                        // 旧形式: Vec<(u64, N)>
                        "entries" => entries = Some(map.next_value::<Vec<(u64, N)>>()?),
                        // 新形式: inds/vals
                        "inds" => inds = Some(map.next_value::<Vec<u32>>()?),
                        "vals" => vals = Some(map.next_value::<Vec<N>>()?),
                        _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
                    }
                }
                let len = len.ok_or_else(|| DeError::missing_field("len"))?;
                let nnz = nnz.ok_or_else(|| DeError::missing_field("nnz"))?;
                // 新形式（inds/vals）があればそれを優先、なければ旧形式（entries）を使う
                if let (Some(inds), Some(vals)) = (inds, vals) {
                    if inds.len() != vals.len() {
                        return Err(DeError::custom("inds and vals length mismatch"));
                    }
                    if inds.len() != nnz {
                        return Err(DeError::custom("nnz does not match inds/vals length"));
                    }
                    let mut vec = ZeroSpVec::with_capacity(nnz);
                    vec.len = len;
                    for i in 0..nnz {
                        unsafe { vec.raw_push(inds[i] as usize, vals[i]) };
                    }
                    Ok(vec)
                } else if let Some(entries) = entries {
                    let mut vec = ZeroSpVec::with_capacity(nnz);
                    vec.len = len;
                    for (index, value) in entries {
                        let idx_u32: u32 = u32::try_from(index).map_err(|_| DeError::custom("index overflow for u32 storage"))?;
                        unsafe { vec.raw_push(idx_u32 as usize, value) };
                    }
                    Ok(vec)
                } else {
                    Err(DeError::custom("missing entries or inds/vals"))
                }
            }
        }
        deserializer.deserialize_struct(
            "ZeroSpVec",
            &["len", "nnz", "inds", "vals", "entries"],
            ZeroSpVecVisitor { marker: std::marker::PhantomData },
        )
    }
}