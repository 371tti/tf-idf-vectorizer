pub mod corpus;
pub mod tfidf;
pub mod token;
pub mod serde;
pub mod evaluate;

use std::{rc::Rc, sync::Arc};
use std::hash::Hash;

use half::f16;
use num_traits::Num;
use ::serde::{Deserialize, Serialize};

use crate::utils::datastruct::map::IndexMap;
use crate::{utils::{datastruct::{vector::{ZeroSpVec, ZeroSpVecTrait}}, normalizer::DeNormalizer}, vectorizer::{corpus::Corpus, tfidf::{DefaultTFIDFEngine, TFIDFEngine}, token::TokenFrequency}};

pub type KeyRc<K> = Rc<K>;

#[derive(Debug, Clone)]
pub struct TFIDFVectorizer<N = f16, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K> + Send + Sync,
    K: Clone + Send + Sync + Eq + std::hash::Hash,
{
    /// Document's TF Vector
    pub documents: IndexMap<KeyRc<K>, TFVector<N>>,
    /// TF Vector's token dimension sample and reverse index
    pub token_dim_rev_index: IndexMap<Box<str>, Vec<KeyRc<K>>>,
    /// Corpus reference
    pub corpus_ref: Arc<Corpus>,
    /// IDF Vector
    pub idf_cache: IDFVector,
    _marker: std::marker::PhantomData<E>,
}

/// 子要素含めて合計64bytesになってる
#[derive(Debug, Serialize, Deserialize, Clone)]
#[repr(align(64))]
pub struct TFVector<N>
where
    N: Num + Copy,
{
    /// TF Vector
    /// use sparse vector
    pub tf_vec: ZeroSpVec<N>,
    /// sum of tokens of this document
    pub token_sum: u64,
    /// denormalize number for this document
    /// for reverse calculation to get token counts from tf values
    pub denormalize_num: f64,
}

// サイズアサーション
// const evaluation より
// エラーなってたら64bytes になってないってこと

#[allow(dead_code)]
const TF_VECTOR_SIZE: usize = core::mem::size_of::<TFVector<f32>>();
static_assertions::const_assert!(TF_VECTOR_SIZE == 64);
#[allow(dead_code)]
const ZSV_SIZE: usize = core::mem::size_of::<ZeroSpVec<f32>>();
static_assertions::const_assert!(ZSV_SIZE == 48);

impl<N> TFVector<N>
where
    N: Num + Copy,
{
    pub fn shrink_to_fit(&mut self) {
        self.tf_vec.shrink_to_fit();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IDFVector
{
    /// IDF Vector it is not sparse because it is mostly filled
    pub idf_vec: Vec<f32>,
    /// denormalize number for idf
    pub denormalize_num: f64,
    /// latest entropy
    pub latest_entropy: u64,
    /// document count
    pub doc_num: u64,
}

impl IDFVector
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

impl <N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K> + Send + Sync,
    K: Clone + Send + Sync + Eq + Hash,
{
    /// Create a new TFIDFVectorizer instance
    pub fn new(corpus_ref: Arc<Corpus>) -> Self {
        let mut instance = Self {
            documents: IndexMap::new(),
            token_dim_rev_index: IndexMap::new(),
            corpus_ref,
            idf_cache: IDFVector::new(),
            _marker: std::marker::PhantomData,
        };
        instance.re_calc_idf();
        instance
    }

    /// set corpus reference
    /// and recalculate idf
    pub fn set_corpus_ref(&mut self, corpus_ref: Arc<Corpus>) {
        self.corpus_ref = corpus_ref;
        self.re_calc_idf();
    }

    /// Corpusに変更があればIDFを再計算する
    pub fn update_idf(&mut self) {
        if self.corpus_ref.get_gen_num() != self.idf_cache.latest_entropy {
            self.re_calc_idf();
        }
        // 更新がなければ何もしない
    }

    /// CorpusからIDFを再計算する
    fn re_calc_idf(&mut self) {
        self.idf_cache.latest_entropy = self.corpus_ref.get_gen_num();
        self.idf_cache.doc_num = self.corpus_ref.get_doc_num();
        (self.idf_cache.idf_vec, self.idf_cache.denormalize_num) = E::idf_vec(&self.corpus_ref, self.token_dim_rev_index.keys());
    }
}

impl <N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K> + Send + Sync,
    K: PartialEq + Clone + Send + Sync + Eq + Hash
{
    /// Add a document
    /// The immediately referenced Corpus is also updated
    pub fn add_doc(&mut self, key: K, doc: &TokenFrequency) {
        // key_rcを作成
        let key_rc = KeyRc::new(key);
        if self.documents.contains_key(&key_rc) {
            self.del_doc(&key_rc);
        }
        let token_sum = doc.token_sum();
        // ドキュメントのトークンをコーパスに追加
        self.add_corpus(doc);
        // 新語彙を差分追加 (O(|doc_vocab|))
        for tok in doc.token_set(){
            self.token_dim_rev_index
                .entry_mut(tok.into_boxed_str())
                .or_insert_with(Vec::new)
                .push(Rc::clone(&key_rc)); // 逆Indexに追加
        }

        let (mut tf_vec, denormalize_num) = E::tf_vec(doc, self.token_dim_rev_index.as_index_set());
        tf_vec.shrink_to_fit();
        let mut doc = TFVector {
            tf_vec,
            token_sum,
            denormalize_num,
        };
        doc.shrink_to_fit();
        self.documents.insert(key_rc, doc);
    }

    pub fn del_doc(&mut self, key: &K)
    where
        K: PartialEq,
    {
        let rc_key = KeyRc::new(key.clone());
        if let Some(doc) = self.documents.get(&rc_key) {
            let tokens = doc.tf_vec.raw_iter()
                .filter_map(|(idx, _)| {
                    let doc_keys = self.token_dim_rev_index.get_with_index_mut(idx);
                    if let Some(doc_keys) = doc_keys {
                        // 逆Indexから削除
                        let rc_key = KeyRc::new(key.clone());
                        doc_keys.retain(|k| *k != rc_key);
                    }
                    let token = self.token_dim_rev_index.get_key_with_index(idx).cloned();
                    token
                }).collect::<Vec<Box<str>>>();
            // ドキュメントを削除
            self.documents.swap_remove(&rc_key);
            // コーパスからも削除
            self.corpus_ref.sub_set(&tokens);
        }
    }

    /// Get TFVector by document ID
    pub fn get_tf(&self, key: &K) -> Option<&TFVector<N>>
    where
        K: Eq + Hash,
    {
        let rc_key = KeyRc::new(key.clone());
        self.documents.get(&rc_key)
    }

    /// Get TokenFrequency by document ID
    /// If quantized, there may be some error
    /// Words not included in the corpus are ignored
    pub fn get_tf_into_token_freq(&self, key: &K) -> Option<TokenFrequency>
    {
        if let Some(tf_vec) = self.get_tf(key) {
            let mut token_freq = TokenFrequency::new();
            tf_vec.tf_vec.raw_iter().for_each(|(idx, val)| {
                if let Some(token) = self.token_dim_rev_index.get_key_with_index(idx) {
                    let val_f64 = (*val).into();
                    let token_num = tf_vec.token_sum.denormalize(tf_vec.denormalize_num) as f64 * val_f64;
                    token_freq.set_token_count(token, token_num as u64);
                } // out of range is ignored
            });
            Some(token_freq)
        } else {
            None
        }
    }

    /// Check if a document with the given ID exists
    pub fn contains_doc(&self, key: &K) -> bool
    where
        K: PartialEq,
    {
        let rc_key = KeyRc::new(key.clone());
        self.documents.contains_key(&rc_key)
    }

    /// Check if the token exists in the token dimension sample
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_dim_rev_index.contains_key(&Box::<str>::from(token))
    }

    /// Check if all tokens in the given TokenFrequency exist in the token dimension sample
    pub fn contains_tokens_from_freq(&self, freq: &TokenFrequency) -> bool {
        freq.token_set_ref_str().iter().all(|tok| self.contains_token(tok))
    }

    pub fn doc_num(&self) -> usize {
        self.documents.len()
    }

    /// add document to corpus
    /// update the referenced corpus
    fn add_corpus(&self, doc: &TokenFrequency) {
        // add document to corpus
        self.corpus_ref.add_set(&doc.token_set_ref_str());
    }
}