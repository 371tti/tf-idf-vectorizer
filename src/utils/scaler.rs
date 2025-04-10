use num::Num;

pub trait AttachedNormalizer{}
impl AttachedNormalizer for u8 {}
impl AttachedNormalizer for u16 {}
impl AttachedNormalizer for u32 {}
impl AttachedNormalizer for u64 {}
impl AttachedNormalizer for u128 {}
impl AttachedNormalizer for i8 {}
impl AttachedNormalizer for i16 {}
impl AttachedNormalizer for i32 {}
impl AttachedNormalizer for i64 {}
impl AttachedNormalizer for i128 {}
impl AttachedNormalizer for isize {}
impl AttachedNormalizer for usize {}
impl AttachedNormalizer for f32 {}
impl AttachedNormalizer for f64 {}

/// 数値をMAX - min もしくは0.0 - 1.0に正規化するためのトレイト
pub trait Normalizer<N> 
where N: Num
{
    /// MAX - 0 もしくは0.0 - 1.0の範囲で変換
    fn into_normalized(self) -> N;
}

impl Normalizer<u16> for f64 {
    fn into_normalized(self) -> u16 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return u16::MAX;
        }
        (self * u16::MAX as f64) as u16
    }
}

impl Normalizer<u8> for f64 {
    fn into_normalized(self) -> u8 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return u8::MAX;
        }
        (self * u8::MAX as f64) as u8
    }
}

impl Normalizer<u32> for f64 {
    fn into_normalized(self) -> u32 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return u32::MAX;
        }
        (self * u32::MAX as f64) as u32
    }
}

impl Normalizer<u64> for f64 {
    fn into_normalized(self) -> u64 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return u64::MAX;
        }
        (self * u64::MAX as f64) as u64
    }
}

impl Normalizer<u128> for f64 {
    fn into_normalized(self) -> u128 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return u128::MAX;
        }
        (self * u128::MAX as f64) as u128
    }
}

impl Normalizer<f32> for f64 {
    fn into_normalized(self) -> f32 {
        if self.is_nan() {
            return 0.0;
        }
        if self < 0.0 {
            return 0.0;
        }
        if self > 1.0 {
            return 1.0;
        }
        self as f32
    }
}

impl Normalizer<f64> for f64 {
    fn into_normalized(self) -> f64 {
        if self.is_nan() {
            return 0.0;
        }
        if self < 0.0 {
            return 0.0;
        }
        if self > 1.0 {
            return 1.0;
        }
        self
    }
}

impl Normalizer<i8> for f64 {
    fn into_normalized(self) -> i8 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return i8::MIN;
        }
        if self > 1.0 {
            return i8::MAX;
        }
        (self * i8::MAX as f64) as i8
    }
}

impl Normalizer<i16> for f64 {
    fn into_normalized(self) -> i16 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return i16::MIN;
        }
        if self > 1.0 {
            return i16::MAX;
        }
        (self * i16::MAX as f64) as i16
    }
}

impl Normalizer<i32> for f64 {
    fn into_normalized(self) -> i32 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return i32::MIN;
        }
        if self > 1.0 {
            return i32::MAX;
        }
        (self * i32::MAX as f64) as i32
    }
}

impl Normalizer<i64> for f64 {
    fn into_normalized(self) -> i64 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return i64::MIN;
        }
        if self > 1.0 {
            return i64::MAX;
        }
        (self * i64::MAX as f64) as i64
    }
}

impl Normalizer<i128> for f64 {
    fn into_normalized(self) -> i128 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return i128::MIN;
        }
        if self > 1.0 {
            return i128::MAX;
        }
        (self * i128::MAX as f64) as i128
    }
}

impl Normalizer<isize> for f64 {
    fn into_normalized(self) -> isize {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return isize::MIN;
        }
        if self > 1.0 {
            return isize::MAX;
        }
        (self * isize::MAX as f64) as isize
    }
}

impl Normalizer<usize> for f64 {
    fn into_normalized(self) -> usize {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return usize::MAX;
        }
        (self * usize::MAX as f64) as usize
    }   
}