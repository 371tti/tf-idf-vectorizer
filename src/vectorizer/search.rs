use num::Num;

use crate::vectorizer::{tfidf::TFIDFEngine, token::TokenFrequency, TFIDFVectorizer};

pub enum Query {
    Dot(TokenFrequency),
    CosineSimilarity(TokenFrequency),
}

pub struct Hits<K, N> 
where 
    N: Num,
{
    pub list: Vec<(K, N)>,
}

impl<'a, N, K, E> TFIDFVectorizer<'a, N, K, E>
where
    N: Num + Copy,
    E: TFIDFEngine<N>,
{
    pub fn search(&self, query: Query) -> Hits<K, N> {
        match query {
            Query::Dot(tf) => SearchWorker::search_dot(self, tf),
            Query::CosineSimilarity(tf) => SearchWorker::search_cosine(self, tf),
        }
    }
}