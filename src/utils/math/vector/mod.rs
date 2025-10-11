// pub mod math;
pub mod serde;

use std::{alloc::{alloc, dealloc, realloc, Layout}, fmt, marker::PhantomData, mem, ptr::{self, NonNull}};
use std::ops::Index;
use std::fmt::Debug;

use num::Num;
/// ZeroSpVecは0要素を疎とした過疎ベクトルを実装です
/// indices と valuesを持ち
/// indicesは要素のインデックスを保持し、
/// valuesは要素の値を保持します
/// 
/// 要素はindicesの昇順でソートされていることを保証します
pub struct ZeroSpVec<N>  
where N: Num
{
    buf: RawZeroSpVec<N>,
    len: usize,
    nnz: usize,
    zero: N,
}

pub trait ZeroSpVecTrait<N>: Clone + Default + Index<usize, Output = N>
where N: Num
{
    unsafe fn ind_ptr(&self) -> *mut u32;
    unsafe fn val_ptr(&self) -> *mut N;
    unsafe fn raw_push(&mut self, index: usize, value: N);
    fn ind_binary_search(&self, index: usize, cut_down: usize) -> Result<usize, usize>;
    fn new() -> Self;
    fn with_capacity(cap: usize) -> Self;
    fn reserve(&mut self, additional: usize);
    fn shrink_to_fit(&mut self);
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn len_mut(&mut self) -> &mut usize;
    fn capacity(&self) -> usize;
    fn nnz(&self) -> usize;
    fn add_dim(&mut self, dim: usize);
    fn clear(&mut self);
    fn push(&mut self, elem: N);
    fn pop(&mut self) -> Option<N>;
    fn get(&self, index: usize) -> Option<&N>;
    fn raw_get(&self, index: usize) -> Option<ValueWithIndex<N>>;
    fn get_with_cut_down(&self, index: usize, cut_down: usize) -> Option<&N>;
    fn raw_get_with_cut_down(&self, index: usize, cut_down: usize) -> Option<ValueWithIndex<N>>;
    fn get_ind(&self, index: usize) -> Option<usize>;
    fn remove(&mut self, index: usize) -> N;
    fn from_vec(vec: Vec<N>) -> Self;
    unsafe fn from_raw_iter(iter: impl Iterator<Item = (usize, N)>, len: usize) -> Self;
    unsafe fn from_sparse_iter(iter: impl Iterator<Item = (usize, N)>, len: usize) -> Self;
    fn iter(&self) -> ZeroSpVecIter<N>;
    fn raw_iter(&self) -> ZeroSpVecRawIter<N>;
}

pub struct ValueWithIndex<'a, N>
where N: Num
{
    pub index: usize,
    pub value: &'a N,
}

impl<N> ZeroSpVecTrait<N> for ZeroSpVec<N> 
where N: Num
{
    #[inline]
    unsafe fn ind_ptr(&self) -> *mut u32 {
        self.buf.ind_ptr.as_ptr()
    }

    #[inline]
    unsafe fn val_ptr(&self) -> *mut N {
        self.buf.val_ptr.as_ptr()
    }

    /// raw_pushは、要素を追加するためのメソッドです。
    /// ただし全体の長さを考慮しませんのであとでlenを更新する必要があります。
    /// 
    /// # Arguments
    /// - `index` - 追加する要素のインデックス
    /// - `value` - 追加する要素の値
    #[inline]
    unsafe fn raw_push(&mut self, index: usize, value: N) {
        if self.nnz == self.buf.cap {
            self.buf.grow();
        }
        unsafe {
            let val_ptr = self.val_ptr().add(self.nnz);
            let ind_ptr = self.ind_ptr().add(self.nnz);
            ptr::write(val_ptr, value);
            debug_assert!(index <= u32::MAX as usize, "index overflow for u32 storage");
            ptr::write(ind_ptr, index as u32);
        }
        self.nnz += 1;
    }

    #[inline]
    fn ind_binary_search(&self, index: usize, cut_down: usize) -> Result<usize, usize> {
        // 要素が無い場合は「まだどこにも挿入されていない」ので Err(0)
        if self.nnz == 0 {
            return Err(0);
        }
        let mut left = cut_down;
        let mut right = self.nnz - 1;
        while left < right {
            let mid = left + (right - left) / 2;
            // read は mid < nnz を満たすため安全
            let mid_index = unsafe { *self.ind_ptr().add(mid) as usize };
            if mid_index == index {
                return Ok(mid);
            } else if mid_index < index {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // ループ終了後 left == right の位置になっている
        let final_index = unsafe { *self.ind_ptr().add(left) as usize };
        if final_index == index {
            Ok(left)
        } else if final_index < index {
            Err(left + 1)
        } else {
            Err(left)
        }
    }

    #[inline]
    fn new() -> Self {
        ZeroSpVec {
            buf: RawZeroSpVec::new(),
            len: 0,
            nnz: 0,
            zero: N::zero(),
        }
    }

    #[inline]
    fn with_capacity(cap: usize) -> Self {
        let mut buf = RawZeroSpVec::new();
        buf.cap = cap;
        buf.cap_set();
        ZeroSpVec {
            buf: buf,
            len: 0,
            nnz: 0,
            zero: N::zero(),
        }
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        let new_cap = self.nnz + additional;
        if new_cap > self.buf.cap {
            self.buf.cap = new_cap;
            self.buf.re_cap_set();
        }
    }

    #[inline]
    fn shrink_to_fit(&mut self) {
        if self.len < self.buf.cap {
            let new_cap = self.nnz;
            self.buf.cap = new_cap;
            self.buf.re_cap_set();
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn len_mut(&mut self) -> &mut usize {
        &mut self.len
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.buf.cap
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn add_dim(&mut self, dim: usize) {
        self.len += dim;
    }

    #[inline]
    fn clear(&mut self) {
        while let Some(_) = self.pop() {
            // do nothing
        }
    }

    #[inline]
    fn push(&mut self, elem: N) {
        if self.nnz == self.buf.cap {
            self.buf.grow();
        }
        if elem != N::zero() {
            unsafe {
                let val_ptr = self.val_ptr().add(self.nnz);
                let ind_ptr = self.ind_ptr().add(self.nnz);
                ptr::write(val_ptr, elem);
                debug_assert!(self.len <= u32::MAX as usize, "index overflow for u32 storage");
                ptr::write(ind_ptr, self.len as u32);
            }
            self.nnz += 1;
        }
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<N> {
        if self.nnz == 0 {
            return None;
        }
        let pop_element = if self.nnz == self.len {
            self.nnz -= 1;
            unsafe {
                Some(ptr::read(self.val_ptr().add(self.nnz)))
            }
        } else {
            Some(N::zero())
        };
        self.len -= 1;
        pop_element
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&N> {
        if index >= self.len {
            return None;
        }
        match self.ind_binary_search(index, 0) {
            Ok(idx) => {
                unsafe {
                    Some(&*self.val_ptr().add(idx))
                }
            },
            Err(_) => {
                Some(&self.zero)
            }
        }
    }

    #[inline]
    fn raw_get(&self, index: usize) -> Option<ValueWithIndex<N>> {
        if index >= self.len {
            return None;
        }
        match self.ind_binary_search(index, 0) {
            Ok(idx) => {
                unsafe {
                    Some(ValueWithIndex {
                        index,
                        value: &*self.val_ptr().add(idx),
                    })
                }
            },
            Err(_) => {
                Some(ValueWithIndex {
                    index,
                    value: &self.zero,
                })
            }
        }
    }

    #[inline]
    fn get_with_cut_down(&self, index: usize, cut_down: usize) -> Option<&N> {
        if index >= self.len {
            return None;
        }
        match self.ind_binary_search(index, cut_down) {
            Ok(idx) => {
                unsafe {
                    Some(&*self.val_ptr().add(idx))
                }
            },
            Err(_) => {
                Some(&self.zero)
            }
        }
    }

    #[inline]
    fn raw_get_with_cut_down(&self, index: usize, cut_down: usize) -> Option<ValueWithIndex<N>> {
        if index >= self.len || cut_down >= self.nnz {
            return None;
        }
        match self.ind_binary_search(index, cut_down) {
            Ok(idx) => {
                unsafe {
                    Some(ValueWithIndex {
                        index,
                        value: &*self.val_ptr().add(idx),
                    })
                }
            },
            Err(_) => {
                Some(ValueWithIndex {
                    index,
                    value: &self.zero,
                })
            }
        }
    }

    #[inline]
    fn get_ind(&self, index: usize) -> Option<usize> {
        if index >= self.nnz {
            return None;
        }
        unsafe {
            Some(ptr::read(self.ind_ptr().add(index)) as usize)
        }
    }



    /// removeメソッド
    /// 
    /// `index` 番目の要素を削除し、削除した要素を返します。
    /// - 論理インデックス `index` が物理的に存在すれば、その値を返す
    /// - 物理的になければ（= デフォルト扱いだった）デフォルト値を返す
    /// 
    /// # Arguments
    /// - `index` - 削除する要素の論理インデックス
    /// 
    /// # Returns
    /// - `N` - 削除した要素の値
    #[inline]
    fn remove(&mut self, index: usize) -> N {
        debug_assert!(index < self.len, "index out of bounds");
        
        // 論理的な要素数は常に1つ減る
        self.len -= 1;

        match self.ind_binary_search(index, 0) {
            Ok(i) => {
                // 今回削除する要素を読みだす
                let removed_val = unsafe {
                    ptr::read(self.val_ptr().add(i))
                };

                // `i` 番目を削除するので、後ろを前にシフト
                let count = self.nnz - i - 1;
                if count > 0 {
                    unsafe {
                        // 値をコピーして前につめる
                        ptr::copy(
                            self.val_ptr().add(i + 1),
                            self.val_ptr().add(i),
                            count
                        );
                        // インデックスもコピーして前につめる
                        ptr::copy(
                            self.ind_ptr().add(i + 1),
                            self.ind_ptr().add(i),
                            count
                        );
                        // シフトした後のインデックスは全て -1 (1つ前に詰める)
                        for offset in i..(self.nnz - 1) {
                            *self.ind_ptr().add(offset) -= 1;
                        }
                    }
                }
                // nnzは 1 減
                self.nnz -= 1;

                // 取り除いた要素を返す
                removed_val
            }
            Err(i) => {
                // index は詰める必要があるので、i 以降の要素のインデックスを -1
                // （たとえば “要素自体は無い” けど、後ろにある要素は
                //  論理インデックスが 1 つ前になる）
                if i < self.nnz {
                    unsafe {
                        for offset in i..self.nnz {
                            *self.ind_ptr().add(offset) -= 1;
                        }
                    }
                }

                // 0返す
                N::zero()
            }
        }
    }

    #[inline]
    fn from_vec(vec: Vec<N>) -> Self {
        let mut zero_sp_vec = ZeroSpVec::with_capacity(vec.len());
        for entry in vec {
            zero_sp_vec.push(entry);
        }
        zero_sp_vec
    }

    // まじでunsafeにするべき
    #[inline]
    unsafe fn from_raw_iter(iter: impl Iterator<Item = (usize, N)>, len: usize) -> Self {
        let mut zero_sp_vec = ZeroSpVec::with_capacity(iter.size_hint().0);
        for (index, value) in iter {
            unsafe {
                zero_sp_vec.raw_push(index, value);
            }
        }
        zero_sp_vec.len = len;
        zero_sp_vec
    }

    /// Build from sparse iterator that yields only non-zero elements (idx, value).
    /// This avoids allocating a full dense Vec when most entries are zero.
    /// unsafeにするべき
    #[inline]
    unsafe fn from_sparse_iter(iter: impl Iterator<Item = (usize, N)>, len: usize) -> Self {
        let mut zero_sp_vec = ZeroSpVec::new();
        for (index, value) in iter {
            if value != N::zero() {
                unsafe {
                    zero_sp_vec.raw_push(index, value);
                }
            }
        }
        zero_sp_vec.len = len;
        zero_sp_vec
    }

    #[inline]
    fn iter(&self) -> ZeroSpVecIter<N> {
        ZeroSpVecIter {
            vec: self,
            pos: 0,
        }
    }

    #[inline]
    fn raw_iter(&self) -> ZeroSpVecRawIter<N> {
        ZeroSpVecRawIter {
            vec: self,
            pos: 0,
        }
    }
}

unsafe impl <N: Num + Send> Send for ZeroSpVec<N> {}
unsafe impl <N: Num + Sync> Sync for ZeroSpVec<N> {}

impl<N> Clone for ZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn clone(&self) -> Self {
        ZeroSpVec {
            buf: self.buf.clone(),
            len: self.len,
            nnz: self.nnz,
            zero: N::zero(),
        }
    }
}

impl<N> Drop for ZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn drop(&mut self) {
        // RawZeroSpVecで実装済み
    }
}

impl<N> Default for ZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn default() -> Self {
        ZeroSpVec::new()
    }
}

impl<N> Index<usize> for ZeroSpVec<N> 
where N: Num
{
    type Output = N;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<N: Num + Debug> Debug for ZeroSpVec<N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.sign_plus() {
            f.debug_struct("DefaultSparseVec")
                .field("buf", &self.buf)
                .field("nnz", &self.nnz)
                .field("len", &self.len)
                .field("zero", &self.zero)
                .finish()
        } else if f.alternate() {
            write!(f, "ZeroSpVec({:?})", self.iter().collect::<Vec<&N>>())
        } else {
            f.debug_list().entries((0..self.len).map(|i| self.get(i).unwrap())).finish()
        }
    }
}

pub struct ZeroSpVecIter<'a, N> 
where N: Num
{
    vec: &'a ZeroSpVec<N>,
    pos: usize,
}

impl<'a, N> Iterator for ZeroSpVecIter<'a, N> 
where N: Num
{
    type Item = &'a N;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.vec.get(self.pos).map(|val| {
            self.pos += 1;
            val
        })
    }
}

pub struct ZeroSpVecRawIter<'a, N> 
where N: Num
{
    vec: &'a ZeroSpVec<N>,
    pos: usize,
}

impl<'a, N> Iterator for ZeroSpVecRawIter<'a, N> 
where N: Num
{
    type Item = (usize, &'a N);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.vec.nnz() {
            let index = unsafe { *self.vec.ind_ptr().add(self.pos) as usize };
            let value = unsafe { &*self.vec.val_ptr().add(self.pos) };
            self.pos += 1;
            Some((index, value))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vec.nnz().saturating_sub(self.pos), Some(self.vec.len()))
    }
}

impl<T> From<Vec<T>> for ZeroSpVec<T>
where T: Num
{
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        ZeroSpVec::from_vec(vec)
    }
}

impl<'a, N> From<ZeroSpVecRawIter<'a, N>> for ZeroSpVec<N>
where
    N: Num + Copy,
{
    #[inline]
    fn from(iter: ZeroSpVecRawIter<'a, N>) -> Self {
        let mut vec = ZeroSpVec::new();
        for (idx, val) in iter {
            unsafe {
                vec.raw_push(idx, *val);
            }
            vec.len += 1;
        }
        vec
    }
}









/// ZeroSpVecの生実装
#[derive(Debug)]
struct RawZeroSpVec<N> 
where N: Num 
{
    val_ptr: NonNull<N>,
    ind_ptr: NonNull<u32>,
    /// cap 定義
    /// 0 => メモリ未確保 (flag)
    /// usize::MAX =>  zero size struct (ZST) として定義 処理の簡略化を実施 (flag)
    /// _ => 実際のcapN
    cap: usize,
    _marker: PhantomData<N>, // 所有権管理用にPhantomDataを追加
}

impl<N> RawZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn new() -> Self {
        // zero size struct (ZST)をusize::MAXと定義 ある種のフラグとして使用
        let cap = if mem::size_of::<N>() == 0 { std::usize::MAX } else { 0 }; 

        RawZeroSpVec {
            // 空のポインタを代入しておく メモリ確保を遅延させる
            val_ptr: NonNull::dangling(),
            // 空のポインタを代入しておく メモリ確保を遅延させる
            ind_ptr: NonNull::dangling(),
            cap: cap,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn grow(&mut self) {
        unsafe {
            let val_elem_size = mem::size_of::<N>();
            let ind_elem_size = mem::size_of::<u32>();

            // 安全性: ZSTの場合growはcapを超えた場合にしか呼ばれない
            // これは必然的にオーバーフローしていることをしめしている
            debug_assert!(val_elem_size != 0, "capacity overflow");

            // アライメントの取得 適切なメモリ確保を行うため
            let t_align = mem::align_of::<N>();
            let usize_align = mem::align_of::<u32>();

            // アロケーション
            let (new_cap, val_ptr, ind_ptr): (usize, *mut N, *mut u32) = 
                if self.cap == 0 {
                    let new_val_layout = Layout::from_size_align(val_elem_size, t_align).expect("Failed to create memory layout");
                    let new_ind_layout = Layout::from_size_align(ind_elem_size, usize_align).expect("Failed to create memory layout");
                    (
                        1,
                        alloc(new_val_layout) as *mut N,
                        alloc(new_ind_layout) as *mut u32,
                    )
                } else {
                    // 効率化: cap * 2 でメモリを確保する 見た目上はO(log n)の増加を実現
                    let new_cap = self.cap * 2;
                    let new_val_layout = Layout::from_size_align(val_elem_size * self.cap, t_align).expect("Failed to create memory layout for reallocation");
                    let new_ind_layout = Layout::from_size_align(ind_elem_size * self.cap, usize_align).expect("Failed to create memory layout for reallocation");
                    (
                        new_cap,
                        realloc(self.val_ptr.as_ptr() as *mut u8, new_val_layout, val_elem_size * new_cap) as *mut N,
                        realloc(self.ind_ptr.as_ptr() as *mut u8, new_ind_layout, ind_elem_size * new_cap) as *mut u32,
                    )
                };

            // アロケーション失敗時の処理
            if val_ptr.is_null() || ind_ptr.is_null() {
                oom();
            }

            // selfに返却
            self.val_ptr = NonNull::new_unchecked(val_ptr);
            self.ind_ptr = NonNull::new_unchecked(ind_ptr);
            self.cap = new_cap;
        }
    }
    
    #[inline]
    fn cap_set(&mut self) {
        unsafe {
            let val_elem_size = mem::size_of::<N>();
            let ind_elem_size = mem::size_of::<u32>();

            let t_align = mem::align_of::<N>();
            let usize_align = mem::align_of::<u32>();

            let new_val_layout = Layout::from_size_align(val_elem_size * self.cap, t_align).expect("Failed to create memory layout");
            let new_ind_layout = Layout::from_size_align(ind_elem_size * self.cap, usize_align).expect("Failed to create memory layout");
            let new_val_ptr = alloc(new_val_layout) as *mut N;
            let new_ind_ptr = alloc(new_ind_layout) as *mut u32;
            if new_val_ptr.is_null() || new_ind_ptr.is_null() {
                oom();
            }
            self.val_ptr = NonNull::new_unchecked(new_val_ptr);
            self.ind_ptr = NonNull::new_unchecked(new_ind_ptr);
        }
    }

    #[inline]
    fn re_cap_set(&mut self) {
        unsafe {
            let val_elem_size = mem::size_of::<N>();
            let ind_elem_size = mem::size_of::<u32>();

            let t_align = mem::align_of::<N>();
            let usize_align = mem::align_of::<u32>();

            let new_val_layout = Layout::from_size_align(val_elem_size * self.cap, t_align).expect("Failed to create memory layout");
            let new_ind_layout = Layout::from_size_align(ind_elem_size * self.cap, usize_align).expect("Failed to create memory layout");
            let new_val_ptr = realloc(self.val_ptr.as_ptr() as *mut u8, new_val_layout, val_elem_size * self.cap) as *mut N;
            let new_ind_ptr = realloc(self.ind_ptr.as_ptr() as *mut u8, new_ind_layout, ind_elem_size * self.cap) as *mut u32;
            if new_val_ptr.is_null() || new_ind_ptr.is_null() {
                oom();
            }
            self.val_ptr = NonNull::new_unchecked(new_val_ptr);
            self.ind_ptr = NonNull::new_unchecked(new_ind_ptr);
        }
    }
}

impl<N> Clone for RawZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            // If cap == 0 (no allocation) or cap == usize::MAX (ZST marker),
            // return a dangling-pointer RawZeroSpVec without allocating.
            if self.cap == 0 || self.cap == usize::MAX {
                return RawZeroSpVec {
                    val_ptr: NonNull::dangling(),
                    ind_ptr: NonNull::dangling(),
                    cap: self.cap,
                    _marker: PhantomData,
                };
            }

            let val_elem_size = mem::size_of::<N>();
            let ind_elem_size = mem::size_of::<u32>();

            let t_align = mem::align_of::<N>();
            let usize_align = mem::align_of::<u32>();

            let new_val_layout = Layout::from_size_align(val_elem_size * self.cap, t_align).expect("Failed to create memory layout");
            let new_ind_layout = Layout::from_size_align(ind_elem_size * self.cap, usize_align).expect("Failed to create memory layout");
            let new_val_ptr = alloc(new_val_layout) as *mut N;
            let new_ind_ptr = alloc(new_ind_layout) as *mut u32;
            if new_val_ptr.is_null() || new_ind_ptr.is_null() {
                oom();
            }
            ptr::copy_nonoverlapping(self.val_ptr.as_ptr(), new_val_ptr, self.cap);
            ptr::copy_nonoverlapping(self.ind_ptr.as_ptr(), new_ind_ptr, self.cap);

            RawZeroSpVec {
                val_ptr: NonNull::new_unchecked(new_val_ptr),
                ind_ptr: NonNull::new_unchecked(new_ind_ptr),
                cap: self.cap,
                _marker: PhantomData,
            }
        }
    }
}

unsafe impl<N: Num + Send> Send for RawZeroSpVec<N> {}
unsafe impl<N: Num + Sync> Sync for RawZeroSpVec<N> {}

impl<N> Drop for RawZeroSpVec<N> 
where N: Num
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // If no allocation was performed (cap == 0) or this is a ZST marker (usize::MAX), skip deallocation.
            if self.cap == 0 || self.cap == usize::MAX {
                return;
            }

            let val_elem_size = mem::size_of::<N>();
            let ind_elem_size = mem::size_of::<u32>();

            let t_align = mem::align_of::<N>();
            let usize_align = mem::align_of::<u32>();

            let new_val_layout = Layout::from_size_align(val_elem_size * self.cap, t_align).expect("Failed to create memory layout");
            let new_ind_layout = Layout::from_size_align(ind_elem_size * self.cap, usize_align).expect("Failed to create memory layout");
            dealloc(self.val_ptr.as_ptr() as *mut u8, new_val_layout);
            dealloc(self.ind_ptr.as_ptr() as *mut u8, new_ind_layout);
        }
    }
}

/// OutOfMemoryへの対処用
/// プロセスを終了させる
/// 本来はpanic!を使用するべきだが、
/// OOMの場合panic!を発生させるとTraceBackによるメモリ仕様が起きてしまうため
/// 仕方なく強制終了させる
/// 本来OOMはOSにより管理され発生前にKillされるはずなのであんまり意味はない。
#[cold]
fn oom() {
    ::std::process::exit(-9999);
}