pub mod corpus;
pub mod tfidf;
pub mod token;
pub mod serde;
pub mod evaluate;

use std::sync::Arc;
use std::hash::Hash;

use indexmap::IndexMap;
use num_traits::Num;
use ::serde::{Deserialize, Serialize};

use crate::{utils::{math::vector::{ZeroSpVec, ZeroSpVecTrait}, normalizer::DeNormalizer}, vectorizer::{corpus::Corpus, tfidf::{DefaultTFIDFEngine, TFIDFEngine}, token::TokenFrequency}};
use ahash::RandomState;

#[derive(Debug, Clone)]
pub struct TFIDFVectorizer<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K> + Send + Sync,
    K: Clone + Send + Sync + Eq + std::hash::Hash,
{
    /// Document's TF Vector
    pub documents: IndexMap<K, TFVector<N, K>, RandomState>,
    /// TF Vector's token dimension sample and reverse index
    pub token_dim_rev_index: IndexMap<Box<str>, Vec<K>, RandomState>,
    /// Corpus reference
    pub corpus_ref: Arc<Corpus>,
    /// IDF Vector
    pub idf_cache: IDFVector<N>,
    _marker: std::marker::PhantomData<E>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TFVector<N, K>
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
    /// Document ID
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IDFVector<N>
where
    N: Num,
{
    /// IDF Vector it is not sparse because it is mostly filled
    pub idf_vec: Vec<N>,
    /// denormalize number for idf
    pub denormalize_num: f64,
    /// latest entropy
    pub latest_entropy: u64,
    /// document count
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

impl <N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K> + Send + Sync,
    K: Clone + Send + Sync + Eq + Hash,
{
    /// Create a new TFIDFVectorizer instance
    pub fn new(corpus_ref: Arc<Corpus>) -> Self {
        let mut instance = Self {
            documents: IndexMap::with_hasher(RandomState::new()),
            token_dim_rev_index: IndexMap::with_hasher(RandomState::new()),
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
        (self.idf_cache.idf_vec, self.idf_cache.denormalize_num) = E::idf_vec(&self.corpus_ref, &self.token_dim_rev_index);
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
        let token_sum = doc.token_sum();
        // ドキュメントのトークンをコーパスに追加
        self.add_corpus(doc);
        // 新語彙を差分追加 (O(|doc_vocab|))
        for tok in doc.token_set_ref_str() {
            if self.token_dim_rev_index.contains_key(tok) {
                self.token_dim_rev_index.get_mut(tok).map(|vec| vec.push(key.clone()));
            } else {
                self.token_dim_rev_index.insert(tok.into(), vec![key.clone()]);
            }
        }

        let (tf_vec, denormalize_num) = E::tf_vec(doc, &self.token_dim_rev_index);
        let mut doc = TFVector {
            tf_vec,
            token_sum,
            denormalize_num,
            key: key,
        };
        doc.shrink_to_fit();
        self.documents.insert(doc.key.clone(), doc);
    }

    pub fn del_doc(&mut self, key: &K)
    where
        K: PartialEq,
    {
        if let Some(doc) = self.documents.get(key) {
            doc.tf_vec.raw_iter()
                .for_each(|(idx, _)| 
                    self.token_dim_rev_index.get_index_mut(idx).iter_mut().for_each(|(s, v)| {
                        // コーパスからドキュメントのトークンを削除
                        self.corpus_ref.sub_set(&[s.as_ref()]);
                        // トークンの逆インデックスからドキュメントを削除
                        v.retain(|doc_key| doc_key != key);
                    })
                );
            // ドキュメントを削除
            self.documents.swap_remove(key);
        }
    }

    /// Get TFVector by document ID
    pub fn get_tf(&self, key: &K) -> Option<&TFVector<N, K>>
    where
        K: Eq + Hash,
    {
        self.documents.get(key)
    }

    /// Get TokenFrequency by document ID
    /// If quantized, there may be some error
    /// Words not included in the corpus are ignored
    pub fn get_tf_into_token_freq(&self, key: &K) -> Option<TokenFrequency>
    {
        if let Some(tf_vec) = self.get_tf(key) {
            let mut token_freq = TokenFrequency::new();
            tf_vec.tf_vec.raw_iter().for_each(|(idx, val)| {
                if let Some((token, _)) = self.token_dim_rev_index.get_index(idx) {
                    let val_f64: f64 = (*val).into();
                    let token_num: f64 = tf_vec.token_sum.denormalize(tf_vec.denormalize_num) * val_f64;
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
        self.documents.contains_key(key)
    }

    /// Check if the token exists in the token dimension sample
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_dim_rev_index.contains_key(token)
    }

    /// Check if all tokens in the given TokenFrequency exist in the token dimension sample
    pub fn contains_tokens_from_freq(&self, freq: &TokenFrequency) -> bool {
        freq.token_set_ref_str().iter().all(|tok| self.token_dim_rev_index.contains_key(*tok))
    }

    pub fn doc_num(&self) -> usize {
        self.documents.len()
    }

    /// add document to corpus
    /// update the referenced corpus
    fn add_corpus(&mut self, doc: &TokenFrequency) {
        // add document to corpus
        self.corpus_ref.add_set(&doc.token_set_ref_str());
    }
}
