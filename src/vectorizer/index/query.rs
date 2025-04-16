use std::ops::{AddAssign, MulAssign};

use num::Num;

use crate::{utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}}, TokenFrequency};

use super::Index;

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded, f64: IntoNormalizer<N> {
    /// query vectorを生成するメソッド
    /// 重要度を考慮せず、トークンの有無だけでベクトルを生成します。
    /// 
    /// # Arguments
    /// * `tokens` - クエリのトークン
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - クエリのベクトル
    pub fn generate_query_mask(&self, tokens: &[&str]) -> ZeroSpVec<N> {
        // TFの計算
        let mut query_tf = TokenFrequency::new();
        query_tf.add_tokens(tokens);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(query_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = if query_tf.contains_token(token) { N::max_normalized() } else { N::zero() }; // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }
        vec
    }

    /// query vectorを生成するメソッド
    /// 
    /// # Arguments
    /// * `tokens` - クエリのトークン
    /// 
    /// # Returns
    /// * `ZeroSpVec<N>` - クエリのベクトル
    #[inline]
    pub fn generate_query(&self, tokens: &[&str]) -> ZeroSpVec<N> {
        // TFの計算
        let mut query_tf = TokenFrequency::new();
        query_tf.add_tokens(tokens);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(query_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = query_tf.tf_token(token); // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }

        let idf_vec = 
        self.corpus_token_freq
            .idf_vector_ref_str::<N>(self.matrix.len() as u64)
            .into_iter()
            .map(|(_, idf)| idf)
            .collect::<Vec<N>>();

        // TF-IDFの計算
        vec = vec.hadamard_normalized_vec(&idf_vec);
        vec
    }
}