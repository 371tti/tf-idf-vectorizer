use num_traits::Num;
use serde::{Serialize, Serializer, Deserialize, Deserializer};
use serde::ser::SerializeStruct;
use crate::utils::datastruct::vector::ZeroSpVecTrait;

use super::ZeroSpVec;

impl<N> Serialize for ZeroSpVec<N>
where
    N: Num + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
    // シリアライズするフィールドは len, nnz, inds, vals
    let mut state = serializer.serialize_struct("ZeroSpVec", 4)?;
        state.serialize_field("len", &(self.len as u64))?;
        state.serialize_field("nnz", &(self.nnz as u64))?;

        // zero-copy
        let inds: &[u32] = unsafe {
            std::slice::from_raw_parts(self.ind_ptr() as *const u32, self.nnz)
        };
        let vals: &[N] = unsafe {
            std::slice::from_raw_parts(self.val_ptr() as *const N, self.nnz)
        };
        state.serialize_field("inds", inds)?;
        state.serialize_field("vals", vals)?;
        state.end()
    }
}

impl<'de, N> Deserialize<'de> for ZeroSpVec<N>
where
    N: Num + Deserialize<'de> + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
    use serde::de::{Visitor, MapAccess, SeqAccess, DeserializeSeed, Error as DeError};
        use std::fmt;

        // シーケンスを直接内部バッファへ書き込むためのSeed
        struct IndsSeed<'a, T: Num + Copy> {
            vec: &'a mut ZeroSpVec<T>,
            count: &'a mut usize,
        }
        impl<'de, 'a, T> DeserializeSeed<'de> for IndsSeed<'a, T>
        where
            T: Num + Deserialize<'de> + Copy,
        {
            type Value = ();
            fn deserialize<Ds>(self, deserializer: Ds) -> Result<Self::Value, Ds::Error>
            where
                Ds: Deserializer<'de>,
            {
                struct V<'b, T: Num + Copy> {
                    vec: &'b mut ZeroSpVec<T>,
                    count: &'b mut usize,
                }
                impl<'de, 'b, T> Visitor<'de> for V<'b, T>
                where
                    T: Num + Deserialize<'de> + Copy,
                {
                    type Value = ();
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        write!(f, "sequence of u32 indices")
                    }
                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: SeqAccess<'de>,
                    {
                        let mut i = 0usize;
                        while let Some(idx) = seq.next_element::<u32>()? {
                            if i == self.vec.buf.cap {
                                self.vec.buf.grow();
                            }
                            unsafe { *self.vec.ind_ptr().add(i) = idx; }
                            i += 1;
                        }
                        *self.count = i;
                        Ok(())
                    }
                }
                deserializer.deserialize_seq(V {
                    vec: self.vec,
                    count: self.count,
                })
            }
        }

        struct ValsSeed<'a, T: Num + Copy> {
            vec: &'a mut ZeroSpVec<T>,
            count: &'a mut usize,
        }
        impl<'de, 'a, T> DeserializeSeed<'de> for ValsSeed<'a, T>
        where
            T: Num + Deserialize<'de> + Copy,
        {
            type Value = ();
            fn deserialize<Ds>(self, deserializer: Ds) -> Result<Self::Value, Ds::Error>
            where
                Ds: Deserializer<'de>,
            {
                struct V<'b, T: Num + Copy> {
                    vec: &'b mut ZeroSpVec<T>,
                    count: &'b mut usize,
                }
                impl<'de, 'b, T> Visitor<'de> for V<'b, T>
                where
                    T: Num + Deserialize<'de> + Copy,
                {
                    type Value = ();
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        write!(f, "sequence of values")
                    }
                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: SeqAccess<'de>,
                    {
                        let mut i = 0usize;
                        while let Some(val) = seq.next_element::<T>()? {
                            if i == self.vec.buf.cap {
                                self.vec.buf.grow();
                            }
                            unsafe { *self.vec.val_ptr().add(i) = val; }
                            i += 1;
                        }
                        *self.count = i;
                        Ok(())
                    }
                }
                deserializer.deserialize_seq(V {
                    vec: self.vec,
                    count: self.count,
                })
            }
        }

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
                let mut len: Option<usize> = None;
                let mut nnz: Option<usize> = None;
                let mut vec = ZeroSpVec::new();
                let mut inds_count: usize = 0;
                let mut vals_count: usize = 0;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "len" => len = Some(map.next_value::<u64>()? as usize),
                        "nnz" => {
                            let v = map.next_value::<u64>()? as usize;
                            if v > vec.buf.cap {
                                let old_cap = vec.buf.cap;
                                vec.buf.cap = v;
                                if old_cap == 0 {
                                    vec.buf.cap_set();
                                } else {
                                    vec.buf.re_cap_set(old_cap);
                                }
                            }
                            nnz = Some(v);
                        },
                        "inds" => { map.next_value_seed(IndsSeed { vec: &mut vec, count: &mut inds_count })?; },
                        "vals" => { map.next_value_seed(ValsSeed { vec: &mut vec, count: &mut vals_count })?; },
                        _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
                    }
                }
                let len = len.ok_or_else(|| DeError::missing_field("len"))?;
                let nnz = nnz.ok_or_else(|| DeError::missing_field("nnz"))?;
                if inds_count != vals_count { return Err(DeError::custom("inds and vals length mismatch")); }
                if inds_count != nnz { return Err(DeError::custom("nnz does not match inds/vals length")); }
                vec.len = len;
                vec.nnz = nnz;
                Ok(vec)
            }

            // bincode等のフィールド名を持たないフォーマット用: フィールド順のシーケンス
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where A: SeqAccess<'de> {
                let len_u64: u64 = seq
                    .next_element()?
                    .ok_or_else(|| DeError::custom("missing len"))?;
                let nnz_u64: u64 = seq
                    .next_element()?
                    .ok_or_else(|| DeError::custom("missing nnz"))?;
                let len = len_u64 as usize;
                let nnz = nnz_u64 as usize;

                // 新形式（inds, vals）のみ対応（bincodeの旧entries形式は非対応）
                // Vecへ一旦受けず、内部バッファへ直接書き込む
                let mut vec = ZeroSpVec::with_capacity(nnz);
                vec.len = len;
                let mut inds_count: usize = 0;
                let mut vals_count: usize = 0;

                seq.next_element_seed(IndsSeed {
                    vec: &mut vec,
                    count: &mut inds_count,
                })?
                .ok_or_else(|| DeError::custom("missing inds"))?;

                seq.next_element_seed(ValsSeed {
                    vec: &mut vec,
                    count: &mut vals_count,
                })?
                .ok_or_else(|| DeError::custom("missing vals"))?;

                if inds_count != vals_count {
                    return Err(DeError::custom("inds and vals length mismatch"));
                }
                if inds_count != nnz {
                    return Err(DeError::custom("nnz does not match inds/vals length"));
                }
                vec.nnz = nnz;
                Ok(vec)
            }
        }
        deserializer.deserialize_struct(
            "ZeroSpVec",
            &["len", "nnz", "inds", "vals"],
            ZeroSpVecVisitor { marker: std::marker::PhantomData },
        )
    }
}