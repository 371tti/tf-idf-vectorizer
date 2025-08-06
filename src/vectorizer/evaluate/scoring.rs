use num::Num;

use crate::vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer};

/// 検索クエリの種類を定義する列挙型
pub enum SimilarityQuery {
    Dot(TokenFrequency),
    CosineSimilarity(TokenFrequency),
    EuclideanDistance(TokenFrequency),
    ManhattanDistance(TokenFrequency),
    ChebyshevDistance(TokenFrequency),
}

/// 検索結果を格納する構造体
pub struct Hits<K> 
{
    pub list: Vec<(K, f64)>,
}

impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    pub fn similarity(&self, query: SimilarityQuery) -> Hits<K> {
        let result = match query {
            SimilarityQuery::Dot(freq) => self.scoring_dot(freq),
            SimilarityQuery::CosineSimilarity(freq) => self.scoring_cosine(freq),
            SimilarityQuery::EuclideanDistance(freq) => self.scoring_euclidean(freq),
            SimilarityQuery::ManhattanDistance(freq) => self.scoring_manhattan(freq),
            SimilarityQuery::ChebyshevDistance(freq) => self.scoring_chebyshev(freq),
        };

    }
}

/// 検索のHL実装
impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    fn scoring_dot(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        // まずtfをTF vector に transform して
        // それから検索域の次元にあわせてIDFを作成
        // クエリのtfidfを計算
        // それを使ってドキュメントのTFIDFとドット積を計算
        // tf1 * tf2 * idf^2 
        let (tf, tf_denormalize_num) = E::tf_vec(&freq, &self.token_dim_sample);
        let (idf, idf_denormalize_num) = E::idf_vec(self.corpus_ref, &self.token_dim_sample);
        let query_tfidf = tf.ha
    }

    fn scoring_cosine(&self, freq: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn scoring_euclidean(&self, freq: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn scoring_manhattan(&self, freq: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn scoring_chebyshev(&self, freq: TokenFrequency) -> Vec<(K, f64)> {

    }
}