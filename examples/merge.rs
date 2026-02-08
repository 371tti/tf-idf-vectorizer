use std::sync::Arc;

use half::f16;
use tf_idf_vectorizer::{Corpus, SimilarityAlgorithm, TFIDFVectorizer, TermFrequency, vectorizer::evaluate::query::Query};

fn main() {
    // build corpus
    let corpus = Arc::new(Corpus::new());

    // make term frequencies
    let mut freq1 = TermFrequency::new();
    freq1.add_terms(&["rust", "高速", "並列", "rust"]);
    let mut freq2 = TermFrequency::new();
    freq2.add_terms(&["rust", "柔軟", "安全", "rust"]);
    let mut freq3 = TermFrequency::new();
    freq3.add_terms(&["rust", "高速", "安全", "抽象"]);

    // add documents to vectorizer
    let mut vectorizer0: TFIDFVectorizer<f16> = TFIDFVectorizer::new(corpus.clone());
    vectorizer0.add_doc("doc1".to_string(), &freq1);
    vectorizer0.add_doc("doc2".to_string(), &freq2);
    vectorizer0.del_doc(&"doc1".to_string());
    vectorizer0.add_doc("doc3".to_string(), &freq1);
    let mut vectorizer1: TFIDFVectorizer<f16> = TFIDFVectorizer::new(corpus.clone());
    vectorizer1.add_doc("doc4".to_string(), &freq3);

    // merge vectorizers
    vectorizer0.merge(vectorizer1);

    let query = Query::and(Query::term("rust"), Query::term("高速"));
    let algorithm = SimilarityAlgorithm::CosineSimilarity;
    let mut result = vectorizer0.search(&algorithm, query);
    result.sort_by_score_desc();

    // print result
    println!("Search Results: \n{}", result);
    // debug
    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer0);
}