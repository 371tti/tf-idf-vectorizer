use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use tf_idf_vectorizer::token::DocumentAnalyzer;
use std::time::Instant;

const MAX_SUDACHI_INPUT_SIZE: usize = 49100;

// 記号や不要なトークンをブラックリストで定義
fn blacklist() -> HashSet<String> {
    [
        ".", ",", "!", "?", ":", ";", "\"", "'", "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\",
        "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~", "`", "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn fetch_text_from_directory(dir_path: &str) -> Vec<(String, String)> {
    let mut articles = Vec::new();
    let dir = Path::new(dir_path);

    for entry in fs::read_dir(dir).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let mut file = File::open(&path).expect("Failed to open file");
            let mut content = String::new();
            file.read_to_string(&mut content).expect("Failed to read file content");
            articles.push((file_name, content));
        }
    }

    articles
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


fn filter_tokens(tokens: Vec<String>, blacklist: &HashSet<String>) -> Vec<String> {
    tokens
        .into_iter()
        .filter(|token| {
            // トークンがブラックリストに含まれていないかつ空白や改行のみではない
            !blacklist.contains(token) && !token.trim().is_empty()
        })
        .collect()
}


fn main() {
    let dir_path = "C:\\RustBuilds\\IDIS\\IDIS_rust\\tf-idf-vectorizer\\popular_wikipedia_articles";
    let articles = fetch_text_from_directory(dir_path);

    let mut analyzer = DocumentAnalyzer::<String>::new();
    let blacklist = blacklist();

    // ファイルの内容をインデックス化
    for (file_name, content) in articles {
        println!("Processing file: {}", file_name);

        let mut tokens = tokenize_with_sudachi(&content, "B");
        tokens.extend(tokenize_with_sudachi(&content, "C"));

        let tokens = filter_tokens(tokens, &blacklist);
        let token_refs: Vec<&str> = tokens.iter().map(AsRef::as_ref).collect();
        analyzer.add_document(file_name, &token_refs, None);
    }


    // インデックスの生成
    println!("Generating index...");
    let index = analyzer.generate_index();

    loop {
        println!("Enter your search query:");
        let mut query = String::new();
        io::stdin().read_line(&mut query).expect("Failed to read line");

        let query_tokens = tokenize_with_sudachi(&query, "C");
        let query_tokens = filter_tokens(query_tokens, &blacklist);
        let query_refs: Vec<&str> = query_tokens.iter().map(AsRef::as_ref).collect();

        println!("Performing search...");

        // 時間測定開始
        let start = Instant::now();

        // 検索処理
        let results = index.search_bm25_tfidf(&query_refs, 100, 1.2, 0.75);

        // 時間測定終了
        let duration = start.elapsed();

        // 結果の表示

        println!("Search results (Time taken: {:.2?}):", duration);
        for (doc_id, similarity) in results {
            println!("Document ID: {}, Similarity: {:.4}", doc_id, similarity);
        }
    }
}
