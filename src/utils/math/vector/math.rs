use std::{cmp::Ordering, ops::{AddAssign, MulAssign}, ptr};

use num::Num;

use super::ZeroSpVec;

impl<N> ZeroSpVec<N> 
where
    N: Num + AddAssign + MulAssign
{
    /// ドット積を計算するメソッド
    ///
    /// # Arguments
    /// * `other` - 他のベクトル
    /// 
    /// # Returns
    /// * `N` - ドット積の結果
    #[inline]
    pub fn dot<R>(&self, other: &Self) -> R
    where
        R: Num + AddAssign,
        N: Into<R> + Copy,
    {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "Vectors must be of the same length to compute dot product."
        );
    
        let mut result = R::zero();
        let self_nnz = self.nnz();
        let other_nnz = other.nnz();
    
        if self_nnz == 0 || other_nnz == 0 {
            return result;
        }
    
        unsafe {
            let self_inds = std::slice::from_raw_parts(self.ind_ptr(), self_nnz);
            let self_vals = std::slice::from_raw_parts(self.val_ptr(), self_nnz);
            let other_inds = std::slice::from_raw_parts(other.ind_ptr(), other_nnz);
            let other_vals = std::slice::from_raw_parts(other.val_ptr(), other_nnz);
            
            let mut i = 0;
            let mut j = 0;
            
            while i < self_nnz && j < other_nnz {
                let s_ind = *self_inds.get_unchecked(i);
                let o_ind = *other_inds.get_unchecked(j);
                match s_ind.cmp(&o_ind) {
                    Ordering::Equal => {
                        result += self_vals[i].into() * other_vals[j].into();
                        i += 1;
                        j += 1;
                    }
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                }
            }
        }
    
        result
    }

    #[inline]
    pub fn norm_sq<R>(&self) -> R
    where
        R: Num + AddAssign + Copy,
        N: Into<R> + Copy,
    {
        let mut result = R::zero();
        let self_nnz = self.nnz();
    
        if self_nnz == 0 {
            return result;
        }
    
        unsafe {
            let self_val_ptr = self.val_ptr();
    
            for i in 0..self_nnz {
                let val: R = (*self_val_ptr.add(i)).into();
                result += val * val;
            }
        }
    
        result
    }

    /// アダマール積を計算するメソッド
    /// 
    /// # Arguments
    /// * `other` - 他のベクトル
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - アダマール積の結果
    #[inline]
    pub fn hadamard(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "Vectors must be of the same length to compute hadamard product."
        );

        let min_nnz = self.nnz().min(other.nnz());
        let mut result: ZeroSpVec<N> = ZeroSpVec::with_capacity(min_nnz);
        result.len = self.len();

        // nnz == 0 ならゼロ埋めで返す
        if self.nnz() == 0 {
            result.len = self.len();
            return result;
        }

        unsafe {
            let mut i = 0;
            let mut j = 0;
            while i < self.nnz() && j < other.nnz() {
                let self_ind = ptr::read(self.ind_ptr().add(i));
                let other_ind = ptr::read(other.ind_ptr().add(j));
                if self_ind == other_ind {
                    // 同じインデックスの要素を掛け算して加算
                    let value = ptr::read(self.val_ptr().add(i)) * ptr::read(other.val_ptr().add(j));
                    result.raw_push(self_ind, value);
                    i += 1;
                    j += 1;
                } else if self_ind < other_ind {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
        result
    }
}