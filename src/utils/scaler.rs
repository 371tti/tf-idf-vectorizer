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
