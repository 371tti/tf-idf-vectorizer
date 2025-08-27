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
        // 内部表現用の一時構造体
        #[derive(Deserialize)]
        struct ZeroSpVecData<N> {
            len: usize,
            nnz: usize,
            entries: Vec<(u64, N)>,
        }

        let data = ZeroSpVecData::deserialize(deserializer)?;

        // capacity は nnz の値で良いとする
        let mut vec = ZeroSpVec::with_capacity(data.nnz);
        vec.len = data.len;
        // entries を内部バッファに移す
        for (index, value) in data.entries {
            unsafe { vec.raw_push(index as usize, value) };
        }
        Ok(vec)
    }
}