pub mod vectorizer;
pub mod utils;

pub use vectorizer::{
    TFIDFVectorizer,
    corpus::Corpus,
    token::TokenFrequency,
    evaluate::scoring::SimilarityQuery,
    evaluate::scoring::Hits,
    serde::TFIDFData,
};