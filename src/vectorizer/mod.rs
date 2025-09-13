pub mod corpus;
pub mod tfidf;
pub mod token;
pub mod serde;
pub mod compute;
pub mod evaluate;

use indexmap::IndexSet;
use num::Num;
use ::serde::{Deserialize, Serialize};

use crate::{utils::{math::vector::{ZeroSpVec, ZeroSpVecTrait}, normalizer::DeNormalizer}, vectorizer::{compute::compare::{Compare, DefaultCompare}, corpus::Corpus, tfidf::{DefaultTFIDFEngine, TFIDFEngine}, token::TokenFrequency}};
use ahash::RandomState;

#[derive(Debug)]
pub struct TFIDFVectorizer<'a, N = f32, K = String, E = DefaultTFIDFEngine, C = DefaultCompare>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
    C: Compare<N>,
{
    /// ドキュメントのTFベクトル
    pub documents: Vec<TFVector<N, K>>,
    /// TFベクトルのトークンの次元サンプル
    pub token_dim_sample: IndexSet<String, RandomState>,
    /// コーパスの参照
    pub corpus_ref: &'a Corpus,
    /// IDFベクトル
    pub idf: IDFVector<N>,
    _marker: std::marker::PhantomData<E>,
    _compare_marker: std::marker::PhantomData<C>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TFVector<N, K>
where
    N: Num + Copy,
{
    /// TFベクトル
    /// スパースベクトルを使用
    pub tf_vec: ZeroSpVec<N>,
    /// トークンの合計数
    pub token_sum: u64,
    /// 正規化解除のための数値
    pub denormalize_num: f64,
    /// ドキュメントID
    pub key: K,
}

impl<N, K> TFVector<N, K>
where
    N: Num + Copy,
{
    pub fn shrink_to_fit(&mut self) {
        self.tf_vec.shrink_to_fit();
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IDFVector<N>
where
    N: Num,
{
    /// IDFベクトル 大体埋まってるのでスパースつかわん
    pub idf_vec: Vec<N>,
    /// 正規化解除のための数値
    pub denormalize_num: f64,
    /// 最新のエントロピー
    pub latest_entropy: u64,
    /// ドキュメント数
    pub doc_num: u64,
}

impl <N> IDFVector<N>
where
    N: Num,
{
    pub fn new() -> Self {
        Self {
            idf_vec: Vec::new(),
            denormalize_num: 1.0,
            latest_entropy: 0,
            doc_num: 0,
        }
    }
}

impl <'a, N, K, E, C> TFIDFVectorizer<'a, N, K, E, C>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
    C: Compare<N>,
{
    /// Create a new TFIDFVectorizer instance
    pub fn new(corpus_ref: &'a Corpus) -> Self {
        let mut instance = Self {
            documents: Vec::new(),
            token_dim_sample: IndexSet::with_hasher(RandomState::new()),
            corpus_ref,
            idf: IDFVector::new(),
            _marker: std::marker::PhantomData,
            _compare_marker: std::marker::PhantomData,
        };
        instance.re_calc_idf();
        instance
    }

    /// Corpusを指定する
    pub fn set_corpus_ref(&mut self, corpus_ref: &'a Corpus) {
        self.corpus_ref = corpus_ref;
        self.re_calc_idf();
    }

    /// Corpusに変更があればIDFを再計算する
    pub fn update_idf(&mut self) {
        if self.corpus_ref.get_gen_num() != self.idf.latest_entropy {
            self.re_calc_idf();
        }
        // 更新がなければ何もしない
    }

    /// CorpusからIDFを再計算する
    fn re_calc_idf(&mut self) {
        self.idf.latest_entropy = self.corpus_ref.get_gen_num();
        self.idf.doc_num = self.corpus_ref.get_doc_num();
        (self.idf.idf_vec, self.idf.denormalize_num) = E::idf_vec(&self.corpus_ref, &self.token_dim_sample)
    }
}

impl <'a, N, K, E, C> TFIDFVectorizer<'a, N, K, E, C>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
    C: Compare<N>,
{
    /// ドキュメントを追加します
    /// 即時参照されているCorpusも更新されます
    pub fn add_doc(&mut self, doc_id: K, doc: &TokenFrequency) {
        let token_sum = doc.token_sum();
        // ドキュメントのトークンをコーパスに追加
        self.add_corpus(doc);
        // 新語彙を差分追加 (O(|doc_vocab|))
        for tok in doc.token_set_ref_str() {
            if !self.token_dim_sample.contains(tok) {
                self.token_dim_sample.insert(tok.to_string());
            }
        }

        let (tf_vec, denormalize_num) = E::tf_vec(doc, &self.token_dim_sample);
        let mut doc = TFVector {
            tf_vec,
            token_sum,
            denormalize_num,
            key: doc_id,
        };
        doc.shrink_to_fit();
        self.documents.push(doc);
    }

    pub fn get_tf(&self, key: &K) -> Option<&TFVector<N, K>>
    where
        K: PartialEq,
    {
        self.documents.iter().find(|doc| &doc.key == key)
    }

    pub fn get_tf_into_token_freq(&self, key: &K) -> Option<TokenFrequency>
    where
        K: PartialEq,
        N: Into<f64>,
    {
        if let Some(tf_vec) = self.get_tf(key) {
            let mut token_freq = TokenFrequency::new();
            tf_vec.tf_vec.raw_iter().for_each(|(idx, val)| {
                if let Some(token) = self.token_dim_sample.get_index(idx) {
                    let val_f64: f64 = (*val).into();
                    let token_num: f64 = tf_vec.token_sum.denormalize(tf_vec.denormalize_num) * val_f64;
                    token_freq.set_token_count(token, token_num as u64);
                } // 範囲外は無視
            });
            Some(token_freq)
        } else {
            None
        }
    }

    /// 参照されてるコーパスを更新
    fn add_corpus(&mut self, doc: &TokenFrequency) {
        // コーパスにドキュメントを追加
        self.corpus_ref.add_set(&doc.token_set_ref_str());
    }
}
