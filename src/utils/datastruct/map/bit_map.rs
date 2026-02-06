use std::ops::{BitAnd, BitOr, BitXor, Not};

pub struct BitMap {
    pub buckets: Vec<L2Bits>,
    pub size: usize,
}

pub struct L2Bits {
    pub bucket_map: u64,
    pub buckets: Box<[L1Bits]>,
}

/// 8 x 64 = 512bits bucket
#[derive(Clone, Copy, Debug, Default)]
#[repr(align(64))]
pub struct L1Bits {
    bucket: [u64; 8],
}

impl BitAnd for &L1Bits {
    type Output = L1Bits;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = L1Bits::default();
        for i in 0..8 {
            result.bucket[i] = self.bucket[i] & rhs.bucket[i];
        }
        result
    }
}

impl BitOr for &L1Bits {
    type Output = L1Bits;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = L1Bits::default();
        for i in 0..8 {
            result.bucket[i] = self.bucket[i] | rhs.bucket[i];
        }
        result
    }
}

impl BitXor for &L1Bits {
    type Output = L1Bits;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = L1Bits::default();
        for i in 0..8 {
            result.bucket[i] = self.bucket[i] ^ rhs.bucket[i];
        }
        result
    }
}

impl Not for &L1Bits {
    type Output = L1Bits;

    fn not(self) -> Self::Output {
        let mut result = L1Bits::default();
        for i in 0..8 {
            result.bucket[i] = !self.bucket[i];
        }
        result
    }
}