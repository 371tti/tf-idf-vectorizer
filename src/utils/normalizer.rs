use half::f16;

pub trait DeNormalizer {
    fn denormalize(&self, denormalize_num: f64) -> f64;
}

impl DeNormalizer for f64 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        self * denormalize_num
    }
}

impl DeNormalizer for f32 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        (*self as f64) * denormalize_num
    }
}

impl DeNormalizer for f16 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const F16_TO_F64: f64 = 1.0 / 65504.0; // f16 max value
        (self.to_f64()) * F16_TO_F64 * denormalize_num
    }
}

impl DeNormalizer for u8 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u8::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u16 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u16::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u32 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u32::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u64 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u64::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}
