use std::{alloc::Layout, iter::FusedIterator, mem, ptr::NonNull};

use num_traits::Num;

#[allow(dead_code)]
const TF_VECTOR_SIZE: usize = core::mem::size_of::<TFVector<u8>>();
static_assertions::const_assert!(TF_VECTOR_SIZE == 32);

pub trait TFVectorTrait<N>
where N: Num + Copy
{
    fn len(&self) -> u32;
    fn nnz(&self) -> u32;
    fn term_sum(&self) -> u32;
    fn new() -> Self;
    fn new_with_capacity(capacity: u32) -> Self;
    fn shrink_to_fit(&mut self);
    fn raw_iter(&self) -> RawTFVectorIter<'_, N>;
    unsafe fn from_vec(ind_vec: Vec<u32>, val_vec: Vec<N>, len: u32, term_sum: u32) -> Self;
    unsafe fn ind_ptr(&self) -> *mut u32;
    unsafe fn val_ptr(&self) -> *mut N;
    /// Power Jump Search
    /// Returns Some((value, sp_vec_raw_ind)) if found, None otherwise
    #[inline(always)]
    unsafe fn power_jump_search(&self, target: u32, start: usize) -> Option<(N, usize)>
    where
        N: Copy,
    {
        let nnz = self.nnz() as usize;
        if start >= nnz {
            return None;
        }

        let ind = unsafe { core::slice::from_raw_parts(self.ind_ptr(), nnz) };
        let val = unsafe { core::slice::from_raw_parts(self.val_ptr(), nnz) };

        // fast path
        let mut lo = start;
        let mut hi = start;

        let s = ind[hi];
        if s == target {
            return Some((val[hi], hi));
        }
        if s > target {
            return None; // forward-only
        }

        // galloping
        let mut step = 1usize;
        loop {
            let next_hi = hi + step;
            if next_hi >= nnz {
                hi = nnz - 1;
                break;
            }
            hi = next_hi;

            if ind[hi] >= target {
                break;
            }

            lo = hi;
            step <<= 1;
        }

        // lower_bound in (lo, hi] => [lo+1, hi+1)
        let mut l = lo + 1;
        let mut r = hi + 1; // exclusive
        while l < r {
            let m = (l + r) >> 1;
            if ind[m] < target {
                l = m + 1;
            } else {
                r = m;
            }
        }

        if l < nnz && ind[l] == target {
            Some((val[l], l))
        } else {
            None
        }
    }
    #[inline(always)]
    fn get_power_jump(&self, target: u32, cut_down: &mut usize) -> Option<N>
    where
        N: Copy,
    {
        unsafe {
            if let Some((v, idx)) = self.power_jump_search(target, *cut_down) {
                *cut_down = idx;
                Some(v)
            } else {
                None
            }
        }
    }
    #[inline(always)]
    fn as_val_slice(&self) -> &[N] {
        unsafe { core::slice::from_raw_parts(self.val_ptr(), self.nnz() as usize) }
    }
    #[inline(always)]
    fn as_ind_slice(&self) -> &[u32] {
        unsafe { core::slice::from_raw_parts(self.ind_ptr(), self.nnz() as usize) }
    }
}

impl<N> TFVectorTrait<N> for TFVector<N> 
where N: Num + Copy
{
    fn new() -> Self {
        Self::low_new()
    }

    #[inline]
    fn new_with_capacity(capacity: u32) -> Self {
        let mut vec = Self::low_new();
        if capacity != 0 {
            vec.set_cap(capacity);
        }
        vec
    }

    #[inline]
    fn shrink_to_fit(&mut self) {
        if self.nnz < self.cap {
            self.set_cap(self.nnz);
        }
    }

    #[inline(always)]
    fn raw_iter(&self) -> RawTFVectorIter<'_, N> {
        RawTFVectorIter {
            vec: self,
            pos: 0,
            end: self.nnz,
        }
    }

    #[inline(always)]
    fn nnz(&self) -> u32 {
        self.nnz
    }

    #[inline(always)]
    fn len(&self) -> u32 {
        self.len
    }

    #[inline(always)]
    fn term_sum(&self) -> u32 {
        self.term_sum
    }

    #[inline(always)]
    unsafe fn from_vec(mut ind_vec: Vec<u32>, mut val_vec: Vec<N>, len: u32, term_sum: u32) -> Self {
        debug_assert_eq!(
            ind_vec.len(),
            val_vec.len(),
            "ind_vec and val_vec must have the same length"
        );

        // sort
        crate::utils::sort::radix_sort_u32_soa(&mut ind_vec, &mut val_vec);

        let nnz = ind_vec.len() as u32;

        if nnz == 0 {
            let mut v = TFVector::low_new();
            v.len = len;
            v.term_sum = term_sum;
            return v;
        }

        // Consume the Vecs and avoid an extra copy:
        // Vec -> Box<[T]> guarantees allocation sized to exactly `len`,
        // which matches `Layout::array::<T>(nnz)` used by `free_alloc()`.
        let inds_box: Box<[u32]> = ind_vec.into_boxed_slice();
        let vals_box: Box<[N]> = val_vec.into_boxed_slice();

        let inds_ptr = Box::into_raw(inds_box) as *mut u32;
        let vals_ptr = Box::into_raw(vals_box) as *mut N;

        TFVector {
            inds: unsafe { NonNull::new_unchecked(inds_ptr) },
            vals: unsafe { NonNull::new_unchecked(vals_ptr) },
            cap: nnz,
            nnz,
            len,
            term_sum,
        }
    }

    #[inline(always)]
    unsafe fn ind_ptr(&self) -> *mut u32 {
        self.inds.as_ptr()
    }

    #[inline(always)]
    unsafe fn val_ptr(&self) -> *mut N {
        self.vals.as_ptr()
    }
}


pub struct RawTFVectorIter<'a, N>
where
    N: Num + 'a,
{
    vec: &'a TFVector<N>,
    pos: u32, // front
    end: u32, // back (exclusive)
}

impl<'a, N> RawTFVectorIter<'a, N>
where
    N: Num + 'a,
{
    #[inline]
    pub fn new(vec: &'a TFVector<N>) -> Self {
        Self { vec, pos: 0, end: vec.nnz }
    }
}

impl<'a, N> Iterator for RawTFVectorIter<'a, N>
where
    N: Num + 'a + Copy,
{
    type Item = (u32, N);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }
        unsafe {
            let i = self.pos as usize;
            self.pos += 1;
            let ind = *self.vec.inds.as_ptr().add(i);
            let val = *self.vec.vals.as_ptr().add(i);
            Some((ind, val))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a, N> DoubleEndedIterator for RawTFVectorIter<'a, N>
where
    N: Num + 'a + Copy,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }
        self.end -= 1;
        unsafe {
            let i = self.end as usize;
            let ind = *self.vec.inds.as_ptr().add(i);
            let val = *self.vec.vals.as_ptr().add(i);
            Some((ind, val))
        }
    }
}

impl<'a, N> ExactSizeIterator for RawTFVectorIter<'a, N>
where
    N: Num + 'a + Copy,
{
    #[inline]
    fn len(&self) -> usize {
        (self.end - self.pos) as usize
    }
}

impl<'a, N> FusedIterator for RawTFVectorIter<'a, N>
where
    N: Num + 'a + Copy,
{}

/// ZeroSpVecの生実装
#[derive(Debug)]
#[repr(align(32))] // どうなんだろうか
pub struct TFVector<N> 
where N: Num
{
    inds: NonNull<u32>,
    vals: NonNull<N>,
    cap: u32,
    nnz: u32,
    len: u32,
    /// sum of terms of this document
    /// denormalize number for this document
    /// for reverse calculation to get term counts from tf values
    term_sum: u32, // for future use
}

/// Low Level Implementation
impl<N> TFVector<N> 
where N: Num
{
    const VAL_SIZE: usize = mem::size_of::<N>();

    #[inline]
    fn low_new() -> Self {
        // ZST は許さん
        debug_assert!(Self::VAL_SIZE != 0, "Zero-sized type is not supported for TFVector");

        TFVector {
            // ダングリングポインタで初期化
            inds: NonNull::dangling(),
            vals: NonNull::dangling(),
            cap: 0,
            nnz: 0,
            len: 0,
            term_sum: 0,
        }
    }


    #[inline]
    #[allow(dead_code)]
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            1
        } else {
            self.cap.checked_mul(2).expect("TFVector capacity overflowed")
        };

        self.set_cap(new_cap);
    }

    #[inline]
    fn set_cap(&mut self, new_cap: u32) {
        if new_cap == 0 {
            // キャパシティを0にする場合はメモリを解放する
            self.free_alloc();
            return;
        }
        let new_inds_layout = Layout::array::<u32>(new_cap as usize).expect("Failed to create inds memory layout");
        let new_vals_layout = Layout::array::<N>(new_cap as usize).expect("Failed to create vals memory layout");

        if self.cap == 0 {
            let new_inds_ptr = unsafe { std::alloc::alloc(new_inds_layout) };
            let new_vals_ptr = unsafe { std::alloc::alloc(new_vals_layout) };
            if new_inds_ptr.is_null() || new_vals_ptr.is_null() {
                if new_inds_ptr.is_null() {
                    oom(new_inds_layout);
                } else {
                    oom(new_vals_layout);
                }
            }

            self.inds = unsafe { NonNull::new_unchecked(new_inds_ptr as *mut u32) };
            self.vals = unsafe { NonNull::new_unchecked(new_vals_ptr as *mut N) };
            self.cap = new_cap;
        } else {
            let old_inds_layout = Layout::array::<u32>(self.cap as usize).expect("Failed to create old inds memory layout");
            let old_vals_layout = Layout::array::<N>(self.cap as usize).expect("Failed to create old vals memory layout");

            let new_inds_ptr = unsafe { std::alloc::realloc(
                self.inds.as_ptr().cast::<u8>(),
                old_inds_layout,
                new_inds_layout.size(),
            ) };
            let new_vals_ptr = unsafe { std::alloc::realloc(
                self.vals.as_ptr().cast::<u8>(),
                old_vals_layout,
                new_vals_layout.size(),
            ) };
            if new_inds_ptr.is_null() || new_vals_ptr.is_null() {
                if new_inds_ptr.is_null() {
                    oom(new_inds_layout);
                } else {
                    oom(new_vals_layout);
                }
            }

            self.inds = unsafe { NonNull::new_unchecked(new_inds_ptr as *mut u32) };
            self.vals = unsafe { NonNull::new_unchecked(new_vals_ptr as *mut N) };
            self.cap = new_cap;
        }
    }

    #[inline]
    fn free_alloc(&mut self) {
        if self.cap != 0 {
            unsafe {
                let inds_layout = Layout::array::<u32>(self.cap as usize).unwrap();
                let vals_layout = Layout::array::<N>(self.cap as usize).unwrap();
                std::alloc::dealloc(self.inds.as_ptr().cast::<u8>(), inds_layout);
                std::alloc::dealloc(self.vals.as_ptr().cast::<u8>(), vals_layout);
            }
        }
        self.inds = NonNull::dangling();
        self.vals = NonNull::dangling();
        self.cap = 0;
    }
}

unsafe impl<N: Num + Send + Sync> Send for TFVector<N> {}
unsafe impl<N: Num + Sync> Sync for TFVector<N> {}

impl<N> Drop for TFVector<N> 
where N: Num
{
    #[inline]
    fn drop(&mut self) {
        self.free_alloc();
    }
}

impl<N> Clone for TFVector<N>
where
    N: Num + Copy,
{
    #[inline]
    fn clone(&self) -> Self {
        let mut new_vec = TFVector::low_new();
        if self.nnz > 0 {
            new_vec.set_cap(self.nnz);
            new_vec.len = self.len;
            new_vec.nnz = self.nnz;
            new_vec.term_sum = self.term_sum;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.inds.as_ptr(),
                    new_vec.inds.as_ptr(),
                    self.nnz as usize,
                );
                std::ptr::copy_nonoverlapping(
                    self.vals.as_ptr(),
                    new_vec.vals.as_ptr(),
                    self.nnz as usize,
                );
            }
        }
        new_vec
    }
}



/// OutOfMemoryへの対処用
/// プロセスを終了させる
/// 本来はpanic!を使用するべきだが、
/// OOMの場合panic!を発生させるとTraceBackによるメモリ仕様が起きてしまうため
/// 仕方なく強制終了させる
/// 本来OOMはOSにより管理され発生前にKillされるはずなのであんまり意味はない。
#[cold]
#[inline(never)]
fn oom(layout: Layout) -> ! {
    std::alloc::handle_alloc_error(layout)
}
