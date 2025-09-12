pub trait DeNormalizer {
    fn denormalize(&self, denormalize_num: f64) -> f64;
}

impl DeNormalizer for f64 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        self * denormalize_num
    }
}

impl DeNormalizer for f32 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        (*self as f64) * denormalize_num
    }
}

impl DeNormalizer for u8 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u8::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u16 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u16::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u32 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u32::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}

impl DeNormalizer for u64 {
    fn denormalize(&self, denormalize_num: f64) -> f64 {
        const DIV_MAX: f64 = 1.0 / (u64::MAX as f64);
        (*self as f64) * DIV_MAX * denormalize_num
    }
}
