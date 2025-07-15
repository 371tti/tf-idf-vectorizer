use tf_idf_vectorizer::vectorizer::{corpus::Corpus, token::TokenFrequency, TFIDFVectorizer};

fn main() {
    let text = "hello world. this is tf-idf-vectorizer lib for rust.
    it is a simple and fast vectorizer.
    you can use it to vectorize your documents and calculate tf-idf scores.";
    let corpus = Corpus::new();
    let mut vectorizer: TFIDFVectorizer<f32> = TFIDFVectorizer::new(&corpus);
    let mut doc = TokenFrequency::new();
    doc.add_tokens(&text.split_whitespace().collect::<Vec<_>>());
    vectorizer.add_doc("doc1".to_string(), &doc);
    vectorizer.update_idf();
    println!("{:?}", vectorizer);
}