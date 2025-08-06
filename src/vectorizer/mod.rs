pub mod corpus;
pub mod tfidf;
pub mod token;
pub mod serde;
pub mod compute;
pub mod evaluate;

use num::Num;
use ::serde::{Deserialize, Serialize};

use crate::{utils::math::vector::ZeroSpVec, vectorizer::{corpus::Corpus, tfidf::{DefaultTFIDFEngine, TFIDFEngine}, token::TokenFrequency}};

#[derive(Debug)]
pub struct TFIDFVectorizer<'a, N = f32, K = String, E = DefaultTFIDFEngine>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// ドキュメントのTFベクトル
    pub documents: Vec<TFVector<N, K>>,
    /// TFベクトルのトークンの次元サンプル
    pub token_dim_sample: Vec<String>,
    /// コーパスの参照
    pub corpus_ref: &'a Corpus,
    /// IDFベクトル
    pub idf: IDFVector<N>,
    _marker: std::marker::PhantomData<E>,
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

impl <'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// Create a new TFIDFVectorizer instance
    pub fn new(corpus_ref: &'a Corpus) -> Self {
        let mut instance = Self {
            documents: Vec::new(),
            token_dim_sample: Vec::new(),
            corpus_ref,
            idf: IDFVector::new(),
            _marker: std::marker::PhantomData,
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

impl <'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    /// ドキュメントを追加します
    /// 即時参照されているCorpusも更新されます
    pub fn add_doc(&mut self, doc_id: K, doc: &TokenFrequency) {
        let token_sum = doc.token_sum();
        // ドキュメントのトークンをコーパスに追加
        self.add_corpus(doc);
        let mut token_set = doc.token_hashset();
        for token in self.token_dim_sample.iter() {
            token_set.remove(token);
        }
        // 新しいトークンを追加
        self.token_dim_sample.extend(token_set.into_iter());

        let (tf_vec, denormalize_num) = E::tf_vec(doc, &self.token_dim_sample);
        let doc = TFVector {
            tf_vec,
            token_sum,
            denormalize_num,
            key: doc_id,
        };
        self.documents.push(doc);
    }

    /// 参照されてるコーパスを更新
    fn add_corpus(&mut self, doc: &TokenFrequency) {
        // コーパスにドキュメントを追加
        self.corpus_ref.add_set(&doc.token_set_ref_str());
    }
}
