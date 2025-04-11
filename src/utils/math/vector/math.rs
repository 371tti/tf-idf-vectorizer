use num::Num;
use rayon::result;

use super::ZeroSpVec;

impl<N> ZeroSpVec<N> 
where
    N: Num
{
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "Vectors must be of the same length to add.");

        let mut result = ZeroSpVec::with_capacity(self.nnz() + other.nnz());
        

        result
    }
}