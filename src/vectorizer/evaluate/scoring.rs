use std::fmt::Debug;

use num::Num;

use crate::{utils::math::vector::{ZeroSpVec, ZeroSpVecTrait}, vectorizer::{compute::compare::{Compare, DefaultCompare}, tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer}};

/// 検索クエリの種類を定義する列挙型
pub enum SimilarityQuery {
    /// 内積
    /// 向きと大きさを考慮した類似度
    Dot(TokenFrequency),
    /// コサイン類似度
    /// 向きのみを考慮した類似度
    CosineSimilarity(TokenFrequency),
    /// ユークリッド距離
    /// ベクトル空間上での直線距離を用いた類似度評価
    /// トークン頻度ベクトル間の距離を計算して、文書間の類似度を測定する
    EuclideanDistance(TokenFrequency),
    /// マンハッタン距離
    /// 各次元の差の絶対値の総和を用いた類似度評価
    /// トークン頻度ベクトル間の距離を計算して、文書間の類似度を測定する
    ManhattanDistance(TokenFrequency),
    /// チェビシェフ距離
    /// 各次元の差の最大値を用いた類似度評価
    /// トークン頻度ベクトル間の距離を計算して、文書間の類似度を測定する
    ChebyshevDistance(TokenFrequency),
}

/// 検索結果を格納する構造体
pub struct Hits<K> 
{
    pub list: Vec<(K, f64)>,
}

impl<K> Hits<K> {
    pub fn new(vec: Vec<(K, f64)>) -> Self {
        Hits { list: vec }
    }

    pub fn sort_by_score(&mut self) -> &mut Self {
    // NaN を除外 (必要なら末尾へ送る運用も可)
    self.list.retain(|(_, s)| !s.is_nan());
    // total_cmp で反射律/推移律を満たす全順序 (NaN 排除済みなので安全)
    self.list.sort_by(|a, b| b.1.total_cmp(&a.1));
    self
    }

    pub fn sort_by_score_rev(&mut self) -> &mut Self{
        // NaN を除外 (必要なら末尾へ送る運用も可)
        self.list.retain(|(_, s)| !s.is_nan());
        // total_cmp で反射律/推移律を満たす全順序 (NaN 排除済みなので安全)
        self.list.sort_by(|a, b| a.1.total_cmp(&b.1));
        self
    }
}

impl<K> Debug for Hits<K>
where
    K: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // Pretty print with alternate formatting: each hit on a new line
            writeln!(f, "Hits [")?;
            for (key, score) in &self.list {
                writeln!(f, "    {:?}: {:.6}", key, score)?;
            }
            write!(f, "]")
        } else {
            // Default debug output
            f.debug_list().entries(&self.list).finish()
        }
    }
}

impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    K: Clone,
    N: Num + Copy,
    E: TFIDFEngine<N>,
    DefaultCompare: Compare<N>,
{
    pub fn similarity(&self, query: SimilarityQuery) -> Hits<K> {
        let result = match query {
            SimilarityQuery::Dot(freq) => self.scoring_dot(freq),
            SimilarityQuery::CosineSimilarity(freq) => self.scoring_cosine(freq),
            SimilarityQuery::EuclideanDistance(freq) => self.scoring_euclidean(freq),
            SimilarityQuery::ManhattanDistance(freq) => self.scoring_manhattan(freq),
            SimilarityQuery::ChebyshevDistance(freq) => self.scoring_chebyshev(freq),
        };

        Hits { list: result }
    }
}

/// 検索のHL実装
impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    K: Clone,
    N: Num + Copy,
    E: TFIDFEngine<N>,
    DefaultCompare: Compare<N>,
{
    fn scoring_dot(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        // まずtfをTF vector に transform して
        // それから検索域の次元にあわせてIDFを作成
        // クエリのtfidfを計算
        // それを使ってドキュメントのTFIDFとドット積を計算
        // tf1 * tf2 * idf^2 
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, query_denorm) =
            E::tfidf_iter_calc(tf.iter().copied(), tf_denormalize_num, idf.iter().copied(), idf_denormalize_num);
        let query_vec: Vec<N> = query_iter.collect();

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, doc_denorm) =
                E::tfidf_iter_calc(doc.tf_vec.iter().copied(), doc.denormalize_num, idf.iter().copied(), idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::dot(query_vec.iter().copied(), doc_iter);
            let score = dot * query_denorm * doc_denorm;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_cosine(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, _query_denorm) =
            E::tfidf_iter_calc_sparse(tf.raw_iter().map(|(idx, val)| (idx, *val)), tf_denormalize_num, &idf, idf_denormalize_num);
        let query_vec: ZeroSpVec<N> = ZeroSpVec::from_raw_iter(query_iter);

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, _doc_denorm) =
                E::tfidf_iter_calc_sparse(doc.tf_vec.raw_iter().map(|(idx, val)| (idx, *val)), doc.denormalize_num, &idf, idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::cosine_similarity(query_vec.raw_iter().map(|(idx, val)| (idx, *val)), doc_iter);
            let score = dot;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_euclidean(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, query_denorm) =
            E::tfidf_iter_calc(tf.iter().copied(), tf_denormalize_num, idf.iter().copied(), idf_denormalize_num);
        let query_vec: Vec<N> = query_iter.collect();

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, doc_denorm) =
                E::tfidf_iter_calc(doc.tf_vec.iter().copied(), doc.denormalize_num, idf.iter().copied(), idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::euclidean_distance(query_vec.iter().copied(), doc_iter);
            let score = dot * query_denorm * doc_denorm;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_manhattan(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, query_denorm) =
            E::tfidf_iter_calc(tf.iter().copied(), tf_denormalize_num, idf.iter().copied(), idf_denormalize_num);
        let query_vec: Vec<N> = query_iter.collect();

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, doc_denorm) =
                E::tfidf_iter_calc(doc.tf_vec.iter().copied(), doc.denormalize_num, idf.iter().copied(), idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::manhattan_distance(query_vec.iter().copied(), doc_iter);
            let score = dot * query_denorm * doc_denorm;
            list.push((doc.key.clone(), score));
        }
        list
    }

    fn scoring_chebyshev(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);

        // 一度 collect して再利用可能にする
        let (query_iter, query_denorm) =
            E::tfidf_iter_calc(tf.iter().copied(), tf_denormalize_num, idf.iter().copied(), idf_denormalize_num);
        let query_vec: Vec<N> = query_iter.collect();

        let mut list = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (doc_iter, doc_denorm) =
                E::tfidf_iter_calc(doc.tf_vec.iter().copied(), doc.denormalize_num, idf.iter().copied(), idf_denormalize_num);

            // Vec からイテレータを再生成して渡す（copied() はプリミティブでほぼコスト無し）
            let dot = DefaultCompare::chebyshev_distance(query_vec.iter().copied(), doc_iter);
            let score = dot * query_denorm * doc_denorm;
            list.push((doc.key.clone(), score));
        }
        list
    }
}