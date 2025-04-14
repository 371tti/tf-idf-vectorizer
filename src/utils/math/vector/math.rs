use std::{ops::{AddAssign, MulAssign}, ptr};

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
    where R: Num + AddAssign, N: Into<R> {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "Vectors must be of the same length to compute dot product."
        );
    
        let mut result: R = R::zero(); // Updated to use R::zero() directly
    
        let self_nnz = self.nnz();
        let other_nnz = other.nnz();
    
        // nnz == 0なら返す
        if self_nnz == 0 {
            return result;
        }
    
        unsafe {
            let mut i = 0;
            let mut j = 0;
            // キャッシュしたポインタを用いる
            let self_ind_ptr = self.ind_ptr();
            let self_val_ptr = self.val_ptr();
            let other_ind_ptr = other.ind_ptr();
            let other_val_ptr = other.val_ptr();
            
            while i < self_nnz && j < other_nnz {
                let self_ind = ptr::read(self_ind_ptr.add(i));
                let other_ind = ptr::read(other_ind_ptr.add(j));
                if self_ind == other_ind {
                    let value = ptr::read(self_val_ptr.add(i)).into() * ptr::read(other_val_ptr.add(j)).into();
                    result += value;
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

    pub fn norm_no_sqrt<R>(&self) -> R
    where R: Num + AddAssign + MulAssign + Copy, N: Into<R> {
        let mut result: R = R::zero(); // Updated to use R::zero() directly
    
        let self_nnz = self.nnz();
    
        // nnz == 0なら返す
        if self_nnz == 0 {
            return result;
        }
    
        unsafe {
            // キャッシュしたポインタを用いる
            let self_val_ptr = self.val_ptr();
            
            for i in 0..self_nnz {
                let value = ptr::read(self_val_ptr.add(i)).into();
                result += value * value;
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