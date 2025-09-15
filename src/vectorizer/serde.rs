use ahash::RandomState;
use indexmap::IndexSet;
use num::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::{vectorizer::{tfidf::{DefaultTFIDFEngine, TFIDFEngine}, IDFVector, TFVector}, Corpus, TFIDFVectorizer};

/// TFIDFVectorizerのデシリアライズ用のデータ構造
/// これは参照を含んでいないため、シリアライズ可能です。
/// `into_tf_idf_vectorizer`メソッドを使用して、`TFIDFVectorizer`に変換できます。
#[derive(Debug, Deserialize, Serialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// ドキュメントのTFベクトル
    pub documents: Vec<TFVector<N, K>>,
    /// TFベクトルのトークンの次元サンプル
    pub token_dim_sample: IndexSet<String, RandomState>,
    /// IDFベクトル
    pub idf: IDFVector<N>,
    #[serde(default, skip_serializing, skip_deserializing)]
    _marker: std::marker::PhantomData<E>,
}

impl<N, K, E> TFIDFData<N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// `TFIDFData`から`TFIDFVectorizer`に変換します。
    /// `corpus_ref`はコーパスの参照です。
    pub fn into_tf_idf_vectorizer<'a>(self, corpus_ref: &'a Corpus) -> TFIDFVectorizer<'a, N, K, E>
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

impl<'a, N, K, E> Serialize for TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy + Serialize,
    K: Serialize,
    E: TFIDFEngine<N>,
{
    /// TFIDFVectorizerをシリアライズします
    /// これは参照を含んでるため、それを除外したものになります。
    /// デシリアライズするには`TFIDFData`を使用してください。
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