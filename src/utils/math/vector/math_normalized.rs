use std::{ops::{AddAssign, MulAssign}, ptr};

use num::Num;

use crate::utils::normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply};

use super::ZeroSpVec;

impl<N> ZeroSpVec<N> 
where
    N: Num + AddAssign + MulAssign + NormalizedMultiply
{
    /// 正規化されたドット積を計算するメソッド
    ///
    /// # Arguments
    /// * `other` - 他のベクトル
    /// 
    /// # Returns
    /// * `N` - ドット積の結果
    #[inline]
    pub fn dot_normalized<R>(&self, other: &Self) -> R
    where R: Num + AddAssign, N: Into<f64> + NormalizedBounded, f64: IntoNormalizer<R> {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "Vectors must be of the same length to compute dot product."
        );
    
        let self_nnz = self.nnz();
        let other_nnz = other.nnz();

        let mut result: R = R::zero();
    
        // nnz == 0なら返す
        if self_nnz == 0 {
            return result;
        }
    
        unsafe {
            let mut i = 0;
            let mut j = 0;
            let mut result_sum: f64 = 0.0;
            // キャッシュしたポインタを用いる
            let self_ind_ptr = self.ind_ptr();
            let self_val_ptr = self.val_ptr();
            let other_ind_ptr = other.ind_ptr();
            let other_val_ptr = other.val_ptr();
            
            while i < self_nnz && j < other_nnz {
                let self_ind = ptr::read(self_ind_ptr.add(i));
                let other_ind = ptr::read(other_ind_ptr.add(j));
                if self_ind == other_ind {
                    let self_val = ptr::read(self_val_ptr.add(i));
                    let other_val = ptr::read(other_val_ptr.add(j));
                    // 正規化された値を計算
                    let value = self_val.mul_normalized(other_val);
                    result_sum += value.into();
                    i += 1;
                    j += 1;
                } else if self_ind < other_ind {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            // 正規化された値を計算
             result = (result_sum / self.len() as f64 / N::max_normalized().into()).into_normalized();
        }
        result
    }

    pub fn cosine_similarity_normalized<R>(&self, other: &Self) -> R
    where R: Num, N: Into<f64>, f64: IntoNormalizer<R> {
        let dot_product: f64 = self.dot(other);
        let self_norm: f64 = self.dot(self).sqrt();
        let other_norm: f64 = other.dot(other).sqrt();
        (dot_product / (self_norm * other_norm)).into_normalized()
    }

    /// 正規化されたアダマール積を計算するメソッド
    /// 
    /// # Arguments
    /// * `other` - 他のベクトル
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - アダマール積の結果
    #[inline]
    pub fn hadamard_normalized(&self, other: &Self) -> Self {
        assert_eq!(
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
                    let self_val = ptr::read(self.val_ptr().add(i));
                    let other_value = ptr::read(other.val_ptr().add(j));
                    result.raw_push(self_ind, self_val.mul_normalized(other_value));
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