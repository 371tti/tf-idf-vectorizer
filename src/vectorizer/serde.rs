use std::sync::Arc;
use std::hash::Hash;

use ahash::RandomState;
use indexmap::IndexMap;
use num_traits::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::{Corpus, TFIDFVectorizer, utils::datastruct::{map::{KeyIndexMap, KeyRc}, vector::ZeroSpVecTrait}, vectorizer::{IDFVector, TFVector, tfidf::{DefaultTFIDFEngine, TFIDFEngine}}};

/// Data structure for deserializing TFIDFVectorizer.
/// This struct does not contain references, so it can be serialized.
/// Use the `into_tf_idf_vectorizer` method to convert to `TFIDFVectorizer`.
#[derive(Debug, Deserialize, Serialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N, K>,
    K: Clone + Eq + Hash,
{
    /// TF vectors for documents
    pub documents: KeyIndexMap<K, TFVector<N>>,
    /// Token dimension sample for TF vectors
    pub token_dim_sample: Vec<Box<str>>,
    /// IDF vector
    pub idf: IDFVector<N>,
    #[serde(default, skip_serializing, skip_deserializing)]
    _marker: std::marker::PhantomData<E>,
}

impl<N, K, E> TFIDFData<N, K, E>
where
    N: Num + Copy + Into<f64> + Send + Sync,
    E: TFIDFEngine<N, K>,
    K: Clone + Send + Sync + Eq + Hash,
{
    /// Convert `TFIDFData` into `TFIDFVectorizer`.
    /// `corpus_ref` is a reference to the corpus.
    pub fn into_tf_idf_vectorizer(self, corpus_ref: Arc<Corpus>) -> TFIDFVectorizer<N, K, E>
    {
        let raw_iter = self.documents.raw_iter();
        let mut token_dim_rev_index: IndexMap<Box<str>, Vec<KeyRc<K>>, RandomState> =
            IndexMap::with_capacity_and_hasher(self.token_dim_sample.len(), RandomState::new());
        self.token_dim_sample.iter().for_each(|token| {
            token_dim_rev_index.insert(token.clone(), Vec::new());
        });
        for (key, doc) in raw_iter {
            doc.tf_vec.raw_iter().for_each(|(idx, _)| {
                let token = &self.token_dim_sample[idx];
                token_dim_rev_index
                    .get_mut(token).unwrap()
                    .push(key.clone());
            });
        }

        let mut instance = TFIDFVectorizer {
            documents: self.documents,
            token_dim_rev_index: token_dim_rev_index,
            corpus_ref,
            idf_cache: self.idf,
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
    E: TFIDFEngine<N, K>,
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
        state.serialize_field("token_dim_sample", &self.token_dim_rev_index.keys().collect::<Vec<_>>())?;
        state.serialize_field("idf", &self.idf_cache)?;
        state.end()
    }
}