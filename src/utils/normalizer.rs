// use num::Num;

// /// 数値をMAX - min もしくは0.0 - 1.0に正規化するためのトレイト
// pub trait IntoNormalizer<N> 
// where N: Num
// {
//     /// MAX - 0 もしくは0.0 - 1.0の範囲で変換
//     fn into_normalized(self) -> N;
// }

// pub trait NormalizedMultiply
// where Self: Num
// {
//     /// 正規化された値を計算するメソッド
//     ///
//     /// # Arguments
//     /// * `self` - 自身の値
//     /// * `other` - 他の値
//     /// 
//     /// # Returns
//     /// * `N` - 正規化された値
//     fn mul_normalized(self, other: Self) -> Self;

// }

// pub trait NormalizedBounded
// where Self: Num
// {
//     /// 正規化された最小値を計算するメソッド
//     ///     
//     /// # Arguments
//     /// * `self` - 自身の値
//     ///
//     /// # Returns
//     /// * `N` - 正規化された最小値
//     fn min_normalized() -> Self;

//     /// 正規化された最大値を計算するメソッド
//     /// 
//     /// # Arguments
//     /// * `self` - 自身の値
//     /// 
//     /// # Returns
//     /// * `N` - 正規化された最大値
//     fn max_normalized() -> Self;
// }

// impl IntoNormalizer<u16> for f64 {
//     #[inline]
//     fn into_normalized(self) -> u16 {
//         (self * u16::MAX as f64) as u16
//     }
// }

// impl IntoNormalizer<u8> for f64 {
//     #[inline]
//     fn into_normalized(self) -> u8 {
//         (self * u8::MAX as f64) as u8
//     }
// }

// impl IntoNormalizer<u32> for f64 {
//     #[inline]
//     fn into_normalized(self) -> u32 {
//         (self * u32::MAX as f64) as u32
//     }
// }

// impl IntoNormalizer<u64> for f64 {
//     #[inline]
//     fn into_normalized(self) -> u64 {
//         (self * u64::MAX as f64) as u64
//     }
// }

// impl IntoNormalizer<u128> for f64 {
//     #[inline]
//     fn into_normalized(self) -> u128 {
//         (self * u128::MAX as f64) as u128
//     }
// }

// impl IntoNormalizer<f32> for f64 {
//     #[inline]
//     fn into_normalized(self) -> f32 {
//         self as f32
//     }
// }

// impl IntoNormalizer<f64> for f64 {
//     #[inline]
//     fn into_normalized(self) -> f64 {
//         self
//     }
// }

// impl IntoNormalizer<i8> for f64 {
//     #[inline]
//     fn into_normalized(self) -> i8 {
//         (self * i8::MAX as f64) as i8
//     }
// }

// impl IntoNormalizer<i16> for f64 {
//     #[inline]
//     fn into_normalized(self) -> i16 {
//         (self * i16::MAX as f64) as i16
//     }
// }

// impl IntoNormalizer<i32> for f64 {
//     #[inline]
//     fn into_normalized(self) -> i32 {
//         (self * i32::MAX as f64) as i32
//     }
// }

// impl IntoNormalizer<i64> for f64 {
//     #[inline]
//     fn into_normalized(self) -> i64 {
//         (self * i64::MAX as f64) as i64
//     }
// }

// impl IntoNormalizer<i128> for f64 {
//     #[inline]
//     fn into_normalized(self) -> i128 {
//         (self * i128::MAX as f64) as i128
//     }
// }

// impl IntoNormalizer<isize> for f64 {
//     #[inline]
//     fn into_normalized(self) -> isize {
//         (self * isize::MAX as f64) as isize
//     }
// }

// impl IntoNormalizer<usize> for f64 {
//     #[inline]
//     fn into_normalized(self) -> usize {
//         (self * usize::MAX as f64) as usize
//     }
// }

// impl NormalizedMultiply for f64 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         self * other
//     }
// }

// impl NormalizedMultiply for f32 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         self * other
//     }
// }

// impl NormalizedMultiply for u8 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         const U8_UNITUP: u16 = u8::MAX as u16 - 1;
//         let prod: u16 = self as u16 * other as u16;
//         ((prod + U8_UNITUP) >> 8) as u8
//     }
// }

// impl NormalizedMultiply for u16 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         const U16_UNITUP: u32 = u16::MAX as u32 - 1;
//         let prod: u32 = self as u32 * other as u32;
//         ((prod + U16_UNITUP) >> 16) as u16
//     }
// }

// impl NormalizedMultiply for u32 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         const U32_UNITUP: u64 = u32::MAX as u64 - 1;
//         let prod: u64 = self as u64 * other as u64;
//         ((prod + U32_UNITUP) >> 32) as u32
//     }
// }

// impl NormalizedMultiply for u64 {
//     #[inline]
//     fn mul_normalized(self, other: Self) -> Self {
//         const U64_UNITUP: u128 = u64::MAX as u128 - 1;
//         let prod: u128 = self as u128 * other as u128;
//         ((prod + U64_UNITUP) >> 64) as u64
//     }
// }

// impl NormalizedBounded for f64 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0.0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         1.0
//     }
// }

// impl NormalizedBounded for f32 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0.0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         1.0
//     }
// }

// impl NormalizedBounded for u8 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         u8::MAX
//     }
// }

// impl NormalizedBounded for u16 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         u16::MAX
//     }
// }

// impl NormalizedBounded for u32 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         u32::MAX
//     }
// }

// impl NormalizedBounded for u64 {
//     #[inline]
//     fn min_normalized() -> Self {
//         0
//     }

//     #[inline]
//     fn max_normalized() -> Self {
//         u64::MAX
//     }
// }