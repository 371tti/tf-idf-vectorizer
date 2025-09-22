use std::sync::Arc;

use ahash::RandomState;
use indexmap::IndexSet;
use num::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::{vectorizer::{tfidf::{DefaultTFIDFEngine, TFIDFEngine}, IDFVector, TFVector}, Corpus, TFIDFVectorizer};

/// Data structure for deserializing TFIDFVectorizer.
/// This struct does not contain references, so it can be serialized.
/// Use the `into_tf_idf_vectorizer` method to convert to `TFIDFVectorizer`.
#[derive(Debug, Deserialize, Serialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// TF vectors for documents
    pub documents: Vec<TFVector<N, K>>,
    /// Token dimension sample for TF vectors
    pub token_dim_sample: IndexSet<Box<str>, RandomState>,
    /// IDF vector
    pub idf: IDFVector<N>,
    #[serde(default, skip_serializing, skip_deserializing)]
    _marker: std::marker::PhantomData<E>,
}

impl<N, K, E> TFIDFData<N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// Convert `TFIDFData` into `TFIDFVectorizer`.
    /// `corpus_ref` is a reference to the corpus.
    pub fn into_tf_idf_vectorizer(self, corpus_ref: Arc<Corpus>) -> TFIDFVectorizer<N, K, E>
    {
        let mut instance = TFIDFVectorizer {
            documents: self.documents,
            token_dim_sample: self.token_dim_sample.clone(),
            corpus_ref,
            idf: self.idf,
            _marker: std::marker::PhantomData,
        };
        instance.update_idf();
        instance
    }
}

impl<N, K, E> Serialize for TFIDFVectorizer<N, K, E>
where
    N: Num + Copy + Serialize,
    K: Serialize,
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
        state.serialize_field("token_dim_sample", &self.token_dim_sample)?;
        state.serialize_field("idf", &self.idf)?;
        state.end()
    }
}