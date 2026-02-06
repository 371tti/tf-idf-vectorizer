use std::sync::Arc;
use std::hash::Hash;

use ahash::RandomState;
use num_traits::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::{Corpus, TFIDFVectorizer, utils::datastruct::{map::IndexMap, vector::TFVectorTrait}, vectorizer::{IDFVector, TFVector, tfidf::{DefaultTFIDFEngine, TFIDFEngine}}};

/// TF-IDF Vectorizer Data Structure (Corpus-free)
///
/// A compact, serializable representation of a TF-IDF vectorizer.
///
/// Unlike [`TFIDFVectorizer`], this struct does **not** hold a `Corpus` reference.
/// It can be converted back into a `TFIDFVectorizer` by providing an `Arc<Corpus>`.
///
/// ### Use Cases
/// - Persistent storage
/// - Network transfer
/// - Memory-efficient snapshots
///
/// ### Serialization
/// Supported.
///
/// ### Deserialization
/// Supported, including internal data expansion.
#[derive(Debug, Deserialize, Serialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
    K: Clone + Eq + Hash,
{
    /// TF vectors for documents
    pub documents: IndexMap<K, TFVector<N>>,
    /// term dimension sample for TF vectors
    pub term_dim_sample: Vec<Box<str>>,
    /// IDF vector
    #[serde(default, skip_serializing, skip_deserializing)]
    pub idf: Option<IDFVector>,
    #[serde(default, skip_serializing, skip_deserializing)]
    pub(crate) _marker: std::marker::PhantomData<E>,
}

impl<N, K, E> TFIDFData<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N>,
    K: Clone + Send + Sync + Eq + Hash,
{
    /// Convert `TFIDFData` into `TFIDFVectorizer`.
    /// `corpus_ref` is a reference to the corpus.
    pub fn into_tf_idf_vectorizer(self, corpus_ref: Arc<Corpus>) -> TFIDFVectorizer<N, K, E>
    {
        let mut term_dim_rev_index: IndexMap<Box<str>, Vec<u32>, RandomState> =
            IndexMap::with_capacity(self.term_dim_sample.len());
        // 順序通りに初めに登録しておく
        self.term_dim_sample.iter().for_each(|term| {
            term_dim_rev_index.insert(term.clone(), Vec::new());
        });
        self.documents.iter().enumerate().for_each(|(id, (_, doc_tf_vec))| {
            doc_tf_vec.raw_iter().for_each(|(idx, _)| {
                term_dim_rev_index.get_with_index_mut(idx as usize).unwrap().push(id as u32);
            });
        });

        let mut instance = TFIDFVectorizer {
            documents: self.documents,
            term_dim_rev_index: term_dim_rev_index,
            corpus_ref,
            idf_cache: IDFVector::new(),
            _marker: std::marker::PhantomData,
        };
        instance.update_idf();
        instance
    }
}

impl<N, K, E> Serialize for TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Serialize + Into<f64> + Send + Sync,
    K: Serialize + Clone + Send + Sync + Eq + Hash,
    E: TFIDFEngine<N>,
{
    /// Serialize TFIDFVectorizer.
    /// This struct contains references, so they are excluded from serialization.
    /// Use `TFIDFData` for deserialization.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TFIDFVectorizer", 3)?;
        state.serialize_field("documents", &self.documents)?;
        state.serialize_field("term_dim_sample", &self.term_dim_rev_index.keys())?;
        state.serialize_field("corpus_ref", &self.corpus_ref)?;
        state.end()
    }
}

impl<'de, N, K, E> Deserialize<'de> for TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Deserialize<'de> + Into<f64> + Send + Sync,
    K: Deserialize<'de> + Clone + Send + Sync + Eq + Hash,
    E: TFIDFEngine<N> + Send + Sync,
{
    /// Deserialize TFIDFVectorizer.
    /// This struct contains references, so they are excluded from deserialization.
    /// Use `TFIDFData` for deserialization.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TFIDFVectorizerHelper<N, K>
        where
            N: Num + Copy,
            K: Clone + Eq + Hash,
        {
            documents: IndexMap<K, TFVector<N>>,
            term_dim_sample: Vec<Box<str>>,
            corpus_ref: Arc<Corpus>,
        }

        let helper = TFIDFVectorizerHelper::<N, K>::deserialize(deserializer)?;
        let mut term_dim_rev_index: IndexMap<Box<str>, Vec<u32>, RandomState> =
            IndexMap::with_capacity(helper.term_dim_sample.len());
        // 順序通りに初めに登録しておく
        helper.term_dim_sample.iter().for_each(|term| {
            term_dim_rev_index.insert(term.clone(), Vec::new());
        });
        helper.documents.iter().enumerate().for_each(|(id, (_, doc_tf_vec))| {
            doc_tf_vec.raw_iter().for_each(|(idx, _)| {
                term_dim_rev_index.get_with_index_mut(idx as usize).unwrap().push(id as u32);
            });
        });

        Ok(TFIDFVectorizer {
            documents: helper.documents,
            term_dim_rev_index,
            corpus_ref: helper.corpus_ref,
            idf_cache: IDFVector::new(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<N, K, E> TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Serialize + Into<f64> + Send + Sync,
    K: Serialize + Clone + Send + Sync + Eq + Hash,
    E: TFIDFEngine<N> + Send + Sync,
{
    pub fn into_tfidf_data(self) -> TFIDFData<N, K, E> {
        let term_dim_sample = self.term_dim_rev_index.keys().clone();
        TFIDFData {
            documents: self.documents,
            term_dim_sample,
            idf: None,
            _marker: std::marker::PhantomData,
        }
    }
}