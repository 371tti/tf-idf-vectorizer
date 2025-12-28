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

    // add documents to vectorizer
    let mut vectorizer: TFIDFVectorizer<f16> = TFIDFVectorizer::new(corpus);    
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);
    vectorizer.del_doc(&"doc1".to_string());
    vectorizer.add_doc("doc3".to_string(), &freq1);

    let query = Query::and(Query::term("rust"), Query::term("安全"));
    let algorithm = SimilarityAlgorithm::CosineSimilarity;
    let mut result = vectorizer.search(&algorithm, query);
    result.sort_by_score_desc();

    // print result
    println!("Search Results: \n{}", result);
    // debug
    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer);
}