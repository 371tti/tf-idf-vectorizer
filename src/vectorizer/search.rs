use num::Num;

use crate::vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer};

/// 検索クエリの種類を定義する列挙型
pub enum Query {
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
    pub fn search(&self, query: Query) -> Hits<K> {
        let result = match query {
            Query::Dot(freq) => self.search_dot(freq),
            Query::CosineSimilarity(freq) => self.search_cosine(freq),
            Query::EuclideanDistance(freq) => self.search_euclidean(freq),
            Query::ManhattanDistance(freq) => self.search_manhattan(freq),
            Query::ChebyshevDistance(freq) => self.search_chebyshev(freq),
        };

    }
}

/// 検索のHL実装
impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    fn search_dot(&self, freq: TokenFrequency) -> Vec<(K, f64)> {
        // まずtfをTF vector に transform して
        // それから検索域の次元にあわせてIDFを作成
        // クエリのtfidfを計算
        // それを使ってドキュメントのTFIDFとドット積を計算
        let tf = E::tf_vec(&freq, &self.token_dim_sample);
        
    }

    fn search_cosine(&self, tf: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn search_euclidean(&self, tf: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn search_manhattan(&self, tf: TokenFrequency) -> Vec<(K, f64)> {

    }

    fn search_chebyshev(&self, tf: TokenFrequency) -> Vec<(K, f64)> {

    }
}