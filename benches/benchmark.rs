use std::collections::HashMap;
use rayon::vec;
use sprs::CsVec;
use tf_idf_vectorizer::token::DocumentAnalyzer;
use reqwest;
use criterion::{criterion_group, criterion_main, Criterion};

fn fetch_text_from_web() -> Result<Vec<String>, reqwest::Error> {
    let urls = vec![
        "https://www.gutenberg.org/files/2600/2600-0.txt", // War and Peace
        "https://www.gutenberg.org/files/1342/1342-0.txt", // Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",     // Alice's Adventures in Wonderland
        "https://www.gutenberg.org/files/84/84-0.txt",     // Frankenstein
        "https://www.gutenberg.org/files/1080/1080-0.txt", // A Modest Proposal
        "https://www.gutenberg.org/files/98/98-0.txt",     // A Tale of Two Cities
        "https://www.gutenberg.org/files/120/120-0.txt",   // Treasure Island
        "https://www.gutenberg.org/files/76/76-0.txt",     // Adventures of Huckleberry Finn
        "https://www.gutenberg.org/files/1400/1400-0.txt", // Great Expectations
        "https://www.gutenberg.org/files/43/43-0.txt",     // The Strange Case of Dr. Jekyll and Mr. Hyde
    ];
    

    let mut texts = Vec::new();

    for url in urls {
        let response = reqwest::blocking::get(url)?;
        let text = response.text()?;
        texts.push(text);
    }

    Ok(texts)
}

fn analyze_and_search_benchmark(c: &mut Criterion) {
    // Fetch text data from the web
    let texts = fetch_text_from_web().expect("Failed to fetch text data");

    // Benchmark DocumentAnalyzer initialization and index generation
    c.bench_function("generate_index", |b| {
        b.iter(|| {
            let mut analyzer = DocumentAnalyzer::<String>::new();

            for (i, text) in texts.iter().enumerate() {
                let tokens: Vec<&str> = text.split_whitespace().collect();
                analyzer.add_document(format!("doc{}", i + 1), &tokens, Some(text));
            }

            analyzer.generate_index()
        });
    });

    // Initialize DocumentAnalyzer and generate the index
    let mut analyzer = DocumentAnalyzer::<String>::new();
    for (i, text) in texts.iter().enumerate() {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        analyzer.add_document(format!("doc{}", i + 1), &tokens, Some(text));
    }
    let index = analyzer.generate_index();

    // Use one of the documents as a query
    let query_tokens: Vec<&str> = vec!["the", "project", "gutenberg", "ebook", "of", "war", "and", "peace", "by", "leo", "tolstoy"];

    // Benchmark the search function
    c.bench_function("search", |b| {
        b.iter(|| {
            index.search(&query_tokens, 10)
        });
    });
}

criterion_group!(benches, analyze_and_search_benchmark);
criterion_main!(benches);


