use std::sync::Arc;

use tf_idf_vectorizer::{Corpus, SimilarityQuery, TFIDFVectorizer, TokenFrequency};

fn main() {
    let corpus = Arc::new(Corpus::new());
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "高速", "並列", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "柔軟", "安全", "rust"]);

    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(corpus);    
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);

    let mut query_tokens = TokenFrequency::new();
    query_tokens.add_tokens(&["rust", "高速"]);
    let query = SimilarityQuery::CosineSimilarity(query_tokens);
    let mut result = vectorizer.similarity(query);
    result.sort_by_score();

    result.list.iter().for_each(|(k, s, l)| {
        println!("doc: {}, score: {}, length: {}", k, s, l);
    });
    println!("result count: {}", result.list.len());
}