use std::collections::HashMap;
use sprs::CsVec;
use tf_idf_vectorizer::token::DocumentAnalyzer;
use reqwest;
use criterion::{criterion_group, criterion_main, Criterion};
use std::io::{self, Write};

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


fn main() {
    // This is for manual runs outside of Criterion benchmarking
    println!("Fetching text data...");
    let texts = fetch_text_from_web().expect("Failed to fetch text data");

    println!("Initializing DocumentAnalyzer...");
    let mut analyzer = DocumentAnalyzer::<String>::new();

    for (i, text) in texts.iter().enumerate() {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        analyzer.add_document(format!("doc{}", i + 1), &tokens, Some(text));
    }

    println!("Generating index...");
    let index = analyzer.generate_index();

    loop {
        println!("Enter your search query:");
        let mut query = String::new();
        io::stdin().read_line(&mut query).expect("Failed to read line");
        let query_tokens: Vec<&str> = query.trim().split_whitespace().collect();
        
        if query_tokens.is_empty() {
            println!("Empty query, exiting...");
            break;
        }

        println!("Performing search...");
        let results = index.search(&query_tokens, 10);

        println!("Search results:");
        for (doc_id, similarity) in results {
            println!("Document ID: {}, Similarity: {:.4}", doc_id, similarity);
        }
    }
    println!("Done.");
}
