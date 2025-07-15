use num::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::vectorizer::{tfidf::{DefaultTFIDFEngine, TFIDFEngine}, IDFVector, TFIDFVectorizer, TFVector};

#[derive(Debug, Deserialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// ドキュメントのTFベクトル
    pub documents: Vec<TFVector<N, K>>,
    /// TFベクトルのトークンの次元サンプル
    pub token_dim_sample: Vec<String>,
    /// IDFベクトル
    pub idf: IDFVector<N>,
    _marker: std::marker::PhantomData<E>,
}

impl TFIDFData {
    pub fn into_tf_idf_vectorizer<'a, N, K, E>(self, corpus_ref: &'a Corpus) -> TFIDFVectorizer<'a, N, K, E>
    where
        N: Num + Copy,
        E: TFIDFEngine<N>,
    {
        TFIDFVectorizer {
            documents: self.documents,
            token_dim_sample: self.token_dim_sample,
            corpus_ref,
            idf: self.idf,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, N, K, E> Serialize for TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy + Serialize,
    K: Serialize,
    E: TFIDFEngine<N>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TFIDFVectorizer", 4)?;
        state.serialize_field("documents", &self.documents)?;
        state.serialize_field("token_dim_sample", &self.token_dim_sample)?;
        state.serialize_field("idf", &self.idf)?;
        state.end()
    }
}