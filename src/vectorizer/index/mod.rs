pub mod try_serde;
pub mod search;
pub mod query;

use std::ops::{AddAssign, MulAssign};

use num::Num;
use serde::Serialize;

use crate::utils::{math::vector::ZeroSpVec, normalizer::{IntoNormalizer, NormalizedBounded, NormalizedMultiply}};

use super::token::TokenFrequency;

/// インデックス
/// ドキュメント単位でインデックスを作成、検索するための構造体です
#[derive(Debug, Clone, Serialize)]
pub struct Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded {
    matrix: Vec<ZeroSpVec<N>>,
    doc_token_count: Vec<u64>,
    doc_id: Vec<String>,
    corpus_token_freq: TokenFrequency,
}

impl<N> Index<N>
where N: Num + Into<f64> + AddAssign + MulAssign + NormalizedMultiply + Copy + NormalizedBounded, f64: IntoNormalizer<N> {
    /// 新しいインデックスを作成するメソッド
    pub fn new() -> Self {
        Self {
            matrix: Vec::new(),
            doc_token_count: Vec::new(),
            doc_id: Vec::new(),
            corpus_token_freq: TokenFrequency::new(),
        }
    }

    /// インデックスのドキュメント数を取得するメソッド
    pub fn doc_num(&self) -> usize {
        self.matrix.len()
    }

    pub fn doc_token_count(&self, index: usize) -> Option<&u64> {
        self.doc_token_count.get(index)
    }

    /// インデックスのコーパス関連のデータ取得、解析のため
    pub fn corpus(&self) -> &TokenFrequency {
        &self.corpus_token_freq
    }

    /// インデックスにドキュメントを追加するメソッド
    /// 
    /// # Arguments
    /// * `doc_id` - ドキュメントのID
    /// * `tokens` - ドキュメントのトークン
    pub fn add_doc(&mut self, doc_id: String, tokens: &[&str]) {
        // TFの計算
        let mut doc_tf = TokenFrequency::new();
        doc_tf.add_tokens(tokens);

        // corpus_token_freqに追加
        let old_corpus_token_num = self.corpus_token_freq.token_num();
        self.corpus_token_freq.add_tokens(tokens);
        let added_corpus_token_num = self.corpus_token_freq.token_num() - old_corpus_token_num;
        self.doc_id.push(doc_id);

        // ZeroSpVecを作成
        let mut vec: ZeroSpVec<N> = ZeroSpVec::with_capacity(doc_tf.token_num());
        for token in self.corpus_token_freq.token_set_ref_str().iter() {
            let tf_val: N = doc_tf.tf_token(token); // ここtf_tokenの内部でいちいちvecのmax計算してるのが最適化されるのか？
            vec.push(tf_val);
        }

        if added_corpus_token_num > 0 {
            // 新しいトークンが追加された場合、matrixを拡張する
            for other_tf in self.matrix.iter_mut() {
                other_tf.add_dim(added_corpus_token_num);
            }
        }
        vec.shrink_to_fit();
        // matrixに追加
        self.doc_token_count.push(doc_tf.token_total_count()); // doc_idのインデックスを追加
        self.matrix.push(vec); // doc_idのインデックスを追加
    }
}