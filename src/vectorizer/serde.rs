use num::Num;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

use crate::vectorizer::{compute::compare::{Compare, DefaultCompare}, corpus::Corpus, tfidf::{DefaultTFIDFEngine, TFIDFEngine}, IDFVector, TFIDFVectorizer, TFVector};

/// TFIDFVectorizerのデシリアライズ用のデータ構造
/// これは参照を含んでいないため、シリアライズ可能です。
/// `into_tf_idf_vectorizer`メソッドを使用して、`TFIDFVectorizer`に変換できます。
#[derive(Debug, Deserialize)]
pub struct TFIDFData<N = f32, K = String, E = DefaultTFIDFEngine, C = DefaultCompare>
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
    _compare_marker: std::marker::PhantomData<C>,
}

impl<N, K, E, C> TFIDFData<N, K, E, C>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
    C: Compare<N>,
{
    /// `TFIDFData`から`TFIDFVectorizer`に変換します。
    /// `corpus_ref`はコーパスの参照です。
    pub fn into_tf_idf_vectorizer<'a>(self, corpus_ref: &'a Corpus) -> TFIDFVectorizer<'a, N, K, E, C>
    {
        let mut instance = TFIDFVectorizer {
            documents: self.documents,
            token_dim_sample: self.token_dim_sample,
            corpus_ref,
            idf: self.idf,
            _marker: std::marker::PhantomData,
            _compare_marker: std::marker::PhantomData,
        };
        instance.update_idf();
        instance
    }
}

impl<'a, N, K, E, C> Serialize for TFIDFVectorizer<'a, N, K, E, C>
where
    N: Num + Copy + Serialize,
    K: Serialize,
    E: TFIDFEngine<N>,
    C: Compare<N>,
{
    /// TFIDFVectorizerをシリアライズします
    /// これは参照を含んでるため、それを除外したものになります。
    /// デシリアライズするには`TFIDFData`を使用してください。
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