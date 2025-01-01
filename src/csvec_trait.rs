use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use sprs::CsVec;


///  FromVec トレイト
pub trait FromVec<T> {
    fn from_vec(vec: Vec<T>) -> Self;
    fn from_vec_parallel(vec: Vec<T>) -> Self;
}

//  CsVec<T> に対する FromVec トレイトの実装
impl<T> FromVec<T> for CsVec<T>
where
    T: Clone + PartialEq + Default + Send + Sync,
{
    /// 通常のfrom_vec実装
    fn from_vec(vec: Vec<T>) -> Self {
        let indices: Vec<usize> = vec
            .iter()
            .enumerate()
            .filter(|&(_, value)| value != &T::default())
            .map(|(index, _)| index)
            .collect();

        let values: Vec<T> = vec
            .iter()
            .filter(|value| **value != T::default())
            .cloned()
            .collect();

        CsVec::new(vec.len(), indices, values)
    }

    /// 並列処理版のfrom_vec実装 順序が保証されない
    fn from_vec_parallel(vec: Vec<T>) -> Self {
        // インデックスと値を並列に収集
        let pairs: Vec<(usize, T)> = vec
            .into_par_iter() // 並列イテレータに変換
            .enumerate()
            .filter(|&(_, ref value)| *value != T::default())
            .collect();

        // インデックスと値を分離
        let (indices, values): (Vec<usize>, Vec<T>) = pairs.into_iter().unzip();

        CsVec::new(indices.len(), indices, values)
    }
}