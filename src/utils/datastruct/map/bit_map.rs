use std::ops::{BitAnd, BitOr, BitXor, Not};

pub struct BitMap {
    pub buckets: Vec<L2Bits>,
    pub size: usize,
}

#[derive(Clone, Debug, Default)]
pub struct L2Bits {
    pub bucket_map: u64,
    pub buckets: Box<[L1Bits]>,
}

impl BitAnd for &L2Bits {
    type Output = L2Bits;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result_buckets = Vec::new();
        let mut result_map = 0u64;

        let mut si = 0usize;
        let mut ri = 0usize;

        for idx in 0..64 {
            let mask = 1u64 << idx;
            let sh = (self.bucket_map & mask) != 0;
            let rh = (rhs.bucket_map & mask) != 0;

            if sh && rh {
                result_map |= mask;
                result_buckets.push(&self.buckets[si] & &rhs.buckets[ri]);
            }
            if sh { si += 1; }
            if rh { ri += 1; }
        }

        L2Bits { bucket_map: result_map, buckets: result_buckets.into_boxed_slice() }
    }
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