use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tf_idf_vectorizer_t::token::DocumentAnalyzer;
use std::time::Instant;

const MAX_SUDACHI_INPUT_SIZE: usize = 49100;
const MAX_FILES_TO_PROCESS: usize = 100000;

// ブラックリスト定義
fn blacklist() -> HashSet<String> {
    let symbols = [
        ".", ",", "!", "?", ":", ";", "\"", "'", "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\",
        "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~", "`", "",
    ];

    let stopwords_english = [
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
        "is", "was", "were", "be", "been", "are", "am", "do", "does", "did", "has", "have", "had",
        // 他の英語のストップワード
    ];
    let stopwords_japanese = [
        "の", "に", "を", "は", "が", "で", "から", "まで", "より", "と", "や", "し", "そして", "けれども",
        "しかし", "だから", "それで", "また", "つまり", "例えば", "なぜ", "どうして", "どの", "どれ", "それ",
        "これ", "あれ", "ここ", "そこ", "あそこ", "どこ", "私", "僕", "俺", "あなた", "彼", "彼女",
        // 他の日本語のストップワード
    ];
    symbols
        .iter()
        .chain(stopwords_english.iter())
        .chain(stopwords_japanese.iter())
        .map(|s| s.to_string())
        .collect()
}

fn split_text_by_bytes(text: &str, max_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_size = 0;

    for ch in text.chars() {
        let ch_size = ch.len_utf8();

        if current_size + ch_size > max_size {
            chunks.push(current_chunk);
            current_chunk = String::new();
            current_size = 0;
        }

        current_chunk.push(ch);
        current_size += ch_size;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

fn tokenize_with_sudachi(text: &str, mode: &str) -> Vec<String> {
    let chunks = split_text_by_bytes(text, MAX_SUDACHI_INPUT_SIZE);
    let mut tokens = Vec::new();

    for chunk in chunks {
        let mut process = Command::new("sudachi")
            .arg("-m")
            .arg(mode) // モード (A/B/C)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start Sudachi");

        {
            let stdin = process.stdin.as_mut().expect("Failed to open stdin");
            stdin
                .write_all(chunk.as_bytes())
                .expect("Failed to write to stdin");
        }

        let output = process.wait_with_output().expect("Failed to read Sudachi output");

        if !output.status.success() {
            eprintln!(
                "Sudachi failed with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        // Sudachiの出力を解析して不要な品詞を除外
        tokens.extend(
            String::from_utf8(output.stdout)
                .expect("Failed to parse Sudachi output")
                .lines()
                .filter_map(|line| {
                    // Sudachiの出力フォーマットをタブ区切りで解析
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() < 2 {
                        return None; // 不正な行は無視
                    }
                    let surface = parts[0]; // トークンの表層形

                    Some(surface.to_string()) // フィルタを通過したトークンを収集
                }),
        );
    }

    tokens
}
// トークンのフィルタリング
fn filter_tokens(tokens: Vec<String>, blacklist: &HashSet<String>) -> Vec<String> {
    tokens
        .into_iter()
        .filter(|token| !blacklist.contains(token) && !token.trim().is_empty())
        .collect()
}

// メイン処理
fn main() {
    let dir_path = "z:\\D\\dev\\web_dev\\idis_v2\\wikipedia_all_articles_fast";
    let blacklist = blacklist();
    let mut analyzer = Arc::new(Mutex::new(DocumentAnalyzer::<String>::new()));
    let file_counter = Arc::new(AtomicUsize::new(0));
    
    // ファイルを逐次処理しつつ並列化
    fs::read_dir(dir_path)
        .expect("Failed to read directory")
        .par_bridge() // Rayonによる並列化
        .try_for_each(|entry| -> Result<(), ()> {
            let entry = entry.expect("Failed to read entry");
            let path = entry.path();

            let file_count = file_counter.load(Ordering::Relaxed);

            if file_count >= MAX_FILES_TO_PROCESS {
            return Err(()); // すでに指定件数を超えている場合は処理をスキップ
            }

            if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let file = File::open(&path).expect("Failed to open file");
            let mut buf_reader = BufReader::new(file);
            let mut content = String::new();

            buf_reader
                .read_to_string(&mut content)
                .expect("Failed to read file");

            println!("Processing file: {} {}", file_count, file_name);
            let mut tokens = tokenize_with_sudachi(&content, "B");
            tokens = filter_tokens(tokens, &blacklist);

            let token_refs: Vec<&str> = tokens.iter().map(AsRef::as_ref).collect();
            let mut analyzer = analyzer.lock().unwrap();
            analyzer.add_document(file_name, &token_refs, None);

            // ファイルカウンターを増加
            file_counter.fetch_add(1, Ordering::Relaxed);
            }

            Ok(())
        }).ok(); // エラーを無視して終了

    // インデックスの生成
    println!("Generating index...");
    let index = analyzer.lock().unwrap().generate_index();
    drop(analyzer);

    loop {
        println!("Enter your search query:");
        let mut query = String::new();
        io::stdin().read_line(&mut query).expect("Failed to read line");

        let query_tokens = tokenize_with_sudachi(&query, "B");
        let query_tokens = filter_tokens(query_tokens, &blacklist);
        let query_refs: Vec<&str> = query_tokens.iter().map(AsRef::as_ref).collect();

        println!("Performing search...");
        let start = Instant::now();
        let results = index.search_bm25_tfidf(&query_refs, 100, 1.5, 0.0 as f64);
        let duration = start.elapsed();

        println!("Search results (Time taken: {:.2?}):", duration);
        for (doc_id, similarity) in results {
            println!("Document ID: {}, Similarity: {:.4}", doc_id, similarity);
        }
    }
}
