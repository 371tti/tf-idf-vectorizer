pub mod corpus;
pub mod tfidf;
pub mod term;
pub mod serde;
pub mod evaluate;

use std::{rc::Rc, sync::Arc};
use std::hash::Hash;

use half::f16;
use num_traits::Num;

use crate::utils::datastruct::vector::{TFVector, TFVectorTrait, IDFVector};
use crate::{DefaultTFIDFEngine, TFIDFEngine, TermFrequency};
use crate::utils::datastruct::map::IndexMap;
use crate::Corpus;

pub type KeyRc<K> = Rc<K>;

/// TF-IDF Vectorizer
///
/// The top-level struct of this crate, providing the main TF-IDF vectorizer features.
///
/// It converts a document collection into TF-IDF vectors and supports similarity
/// computation and search functionality.
///
/// ### Internals
/// - Corpus vocabulary
/// - Sparse TF vectors per document
/// - term index mapping
/// - Cached IDF vector
/// - Pluggable TF-IDF engine
/// - Inverted document index
///
/// ### Type Parameters
/// - `N`: Vector parameter type (e.g., `f32`, `f64`, `u16`)
/// - `K`: Document key type (e.g., `String`, `usize`)
/// - `E`: TF-IDF calculation engine
///
/// ### Notes
/// - Requires an `Arc<Corpus>` on construction
/// - `Corpus` can be shared across multiple vectorizers
///
/// ### Serialization
/// Supported.  
/// Serialized data includes the `Corpus` reference.
///
/// For corpus-independent storage, use [`TFIDFData`].
#[derive(Debug, Clone)]
pub struct TFIDFVectorizer<N = f16, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
    K: Clone + Send + Sync + Eq + std::hash::Hash,
{
    /// Document's TF Vector
    pub documents: IndexMap<KeyRc<K>, TFVector<N>>,
    /// TF Vector's term dimension sample and reverse index
    pub term_dim_rev_index: IndexMap<Box<str>, Vec<KeyRc<K>>>,
    /// Corpus reference
    pub corpus_ref: Arc<Corpus>,
    /// IDF Vector
    pub idf_cache: IDFVector,
    _marker: std::marker::PhantomData<E>,
}

impl <N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
    K: Clone + Send + Sync + Eq + Hash,
{
    /// Create a new TFIDFVectorizer instance
    pub fn new(corpus_ref: Arc<Corpus>) -> Self {
        let mut instance = Self {
            documents: IndexMap::new(),
            term_dim_rev_index: IndexMap::new(),
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
        self.idf_cache.idf_vec = E::idf_vec(&self.corpus_ref, self.term_dim_rev_index.keys());
    }
}

impl <N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N> + Send + Sync,
    K: PartialEq + Clone + Send + Sync + Eq + Hash
{
    /// Add a document
    /// The immediately referenced Corpus is also updated
    pub fn add_doc(&mut self, key: K, doc: &TermFrequency) {
        // key_rcを作成
        let key_rc = KeyRc::new(key);
        if self.documents.contains_key(&key_rc) {
            self.del_doc(&key_rc);
        }
        // ドキュメントのトークンをコーパスに追加
        self.add_corpus(doc);
        // 新語彙を差分追加 (O(|doc_vocab|))
        for tok in doc.term_set(){
            self.term_dim_rev_index
                .entry_mut(tok.into_boxed_str())
                .or_insert_with(Vec::new)
                .push(Rc::clone(&key_rc)); // 逆Indexに追加
        }

        let tf_vec= E::tf_vec(doc, self.term_dim_rev_index.as_index_set());
        self.documents.insert(key_rc, tf_vec);
    }

    pub fn del_doc(&mut self, key: &K)
    where
        K: PartialEq,
    {
        let rc_key = KeyRc::new(key.clone());
        if let Some(tf_vec) = self.documents.get(&rc_key) {
            let terms = tf_vec.raw_iter()
                .filter_map(|(idx, _)| {
                    let idx = idx as usize;
                    let doc_keys = self.term_dim_rev_index.get_with_index_mut(idx);
                    if let Some(doc_keys) = doc_keys {
                        // 逆Indexから削除
                        let rc_key = KeyRc::new(key.clone());
                        doc_keys.retain(|k| *k != rc_key);
                    }
                    let term = self.term_dim_rev_index.get_key_with_index(idx).cloned();
                    term
                }).collect::<Vec<Box<str>>>();
            // ドキュメントを削除
            self.documents.swap_remove(&rc_key);
            // コーパスからも削除
            self.corpus_ref.sub_set(&terms);
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

    /// Get TermFrequency by document ID
    /// If quantized, there may be some error
    /// Words not included in the corpus are ignored
    pub fn get_tf_into_term_freq(&self, key: &K) -> Option<TermFrequency>
    {
        if let Some(tf_vec) = self.get_tf(key) {
            let mut term_freq = TermFrequency::new();
            tf_vec.raw_iter().for_each(|(idx, val)| {
                let idx = idx as usize;
                if let Some(term) = self.term_dim_rev_index.get_key_with_index(idx) {
                    let term_num = E::tf_denorm(val);
                    term_freq.set_term_count(term, term_num as u64);
                } // out of range is ignored
            });
            Some(term_freq)
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

    /// Check if the term exists in the term dimension sample
    pub fn contains_term(&self, term: &str) -> bool {
        self.term_dim_rev_index.contains_key(&Box::<str>::from(term))
    }

    /// Check if all terms in the given TermFrequency exist in the term dimension sample
    pub fn contains_terms_from_freq(&self, freq: &TermFrequency) -> bool {
        freq.term_set_ref_str().iter().all(|tok| self.contains_term(tok))
    }

    pub fn doc_num(&self) -> usize {
        self.documents.len()
    }

    /// add document to corpus
    /// update the referenced corpus
    fn add_corpus(&self, doc: &TermFrequency) {
        // add document to corpus
        self.corpus_ref.add_set(&doc.term_set_ref_str());
    }
}