use num_traits::Num;
use serde::{Deserialize, Serialize, Serializer};
use serde::ser::SerializeStruct;

use crate::utils::datastruct::vector::{TFVector, TFVectorTrait};

impl<N> Serialize for TFVector<N>
where
    N: Num + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        // シリアライズするフィールドは len, nnz, inds, vals, term_sum
        let mut state = serializer.serialize_struct("TFVector", 5)?;
        let vals = self.as_val_slice();
        let inds = self.as_ind_slice();
        state.serialize_field("len", &(self.len()))?;
        state.serialize_field("nnz", &(self.nnz()))?;
        state.serialize_field("inds", inds)?;
        state.serialize_field("vals", vals)?;
        state.serialize_field("term_sum", &self.term_sum())?;
        state.end()
    }
}

#[derive(Deserialize)]
struct TFVectorDe<N> {
    len: u32,
    #[serde(default)]
    nnz: Option<u32>,
    inds: Vec<u32>,
    vals: Vec<N>,
    term_sum: u32,
}

impl<'de, N> Deserialize<'de> for TFVector<N>
where
    N: Num + Deserialize<'de> + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let de = TFVectorDe::<N>::deserialize(deserializer)?;

        debug_assert_eq!(de.inds.len(), de.vals.len());
        debug_assert!(de.nnz.map_or(true, |n| n as usize == de.inds.len()));

        Ok(unsafe { TFVector::from_vec(de.inds, de.vals, de.len, de.term_sum) })
    }
}