use half::f16;

pub trait DeNormalizer {
    fn denormalize(&self, denormalize_num: f64) -> f32;
}

impl DeNormalizer for f64 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        (self * denormalize_num) as f32
    }
}

impl DeNormalizer for f32 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        (*self * denormalize_num as f32) as f32
    }
}

impl DeNormalizer for f16 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        (self.to_f32()) * denormalize_num as f32
    }
}

impl DeNormalizer for u8 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        const DIV_MAX: f32 = 1.0 / (u8::MAX as f32);
        (*self as f32) * DIV_MAX * denormalize_num as f32
    }
}

impl DeNormalizer for u16 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        const DIV_MAX: f32 = 1.0 / (u16::MAX as f32);
        (*self as f32) * DIV_MAX * denormalize_num as f32
    }
}

impl DeNormalizer for u32 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        const DIV_MAX: f32 = 1.0 / (u32::MAX as f32);
        (*self as f32) * DIV_MAX * denormalize_num as f32
    }
}

impl DeNormalizer for u64 {
    #[inline]
    fn denormalize(&self, denormalize_num: f64) -> f32 {
        const DIV_MAX: f32 = 1.0 / (u64::MAX as f32);
        (*self as f32) * DIV_MAX * denormalize_num as f32
    }
}
