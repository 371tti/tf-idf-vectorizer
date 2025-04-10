use num::Num;
/// 数値をMAX - min もしくは0.0 - 1.0に正規化するためのトレイト
pub trait Normalizer<T> 
where T: Num
{
    /// MAX - min もしくは0.0 - 1.0の範囲で変換
    fn into_normalized(self) -> T;
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
            return 65535;
        }
        (self * 65535.0) as u16
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
            return 255;
        }
        (self * 255.0) as u8
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
            return 4294967295;
        }
        (self * 4294967295.0) as u32
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
            return 18446744073709551615;
        }
        (self * 18446744073709551615.0) as u64
    }
}

impl Normalizer<u64> for f32 {
    fn into_normalized(self) -> u64 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return 18446744073709551615;
        }
        (self * 18446744073709551615.0) as u64
    }
}

impl Normalizer<u32> for f32 {
    fn into_normalized(self) -> u32 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return 4294967295;
        }
        (self * 4294967295.0) as u32
    }
    
}

impl Normalizer<u16> for f32 {
    fn into_normalized(self) -> u16 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return 65535;
        }
        (self * 65535.0) as u16
    }
}

impl Normalizer<u8> for f32 {
    fn into_normalized(self) -> u8 {
        if self.is_nan() {
            return 0;
        }
        if self < 0.0 {
            return 0;
        }
        if self > 1.0 {
            return 255;
        }
        (self * 255.0) as u8
    }
}

impl Normalizer<f64> for u64 {
    fn into_normalized(self) -> f64 {
        if self == 0 {
            return 0.0;
        }
        if self > 18446744073709551615 {
            return 1.0;
        }
        (self as f64) / 18446744073709551615.0
    }
}

impl Normalizer<f64> for u32 {
    fn into_normalized(self) -> f64 {
        if self == 0 {
            return 0.0;
        }
        if self > 4294967295 {
            return 1.0;
        }
        (self as f64) / 4294967295.0
    }
}

impl Normalizer<f64> for u16 {
    fn into_normalized(self) -> f64 {
        if self == 0 {
            return 0.0;
        }
        if self > 65535 {
            return 1.0;
        }
        (self as f64) / 65535.0
    }
}

impl Normalizer<f64> for u8 {
    fn into_normalized(self) -> f64 {
        if self == 0 {
            return 0.0;
        }
        if self > 255 {
            return 1.0;
        }
        (self as f64) / 255.0
    }
}

impl Normalizer<f32> for u64 {
    fn into_normalized(self) -> f32 {
        if self == 0 {
            return 0.0;
        }
        if self > 18446744073709551615 {
            return 1.0;
        }
        (self as f32) / 18446744073709551615.0
    }
}

impl Normalizer<f32> for u32 {
    fn into_normalized(self) -> f32 {
        if self == 0 {
            return 0.0;
        }
        if self > 4294967295 {
            return 1.0;
        }
        (self as f32) / 4294967295.0
    }
}

impl Normalizer<f32> for u16 {
    fn into_normalized(self) -> f32 {
        if self == 0 {
            return 0.0;
        }
        if self > 65535 {
            return 1.0;
        }
        (self as f32) / 65535.0
    }
}

impl Normalizer<f32> for u8 {
    fn into_normalized(self) -> f32 {
        if self == 0 {
            return 0.0;
        }
        if self > 255 {
            return 1.0;
        }
        (self as f32) / 255.0
    }
}