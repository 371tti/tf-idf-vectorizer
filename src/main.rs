
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::iter::{ParallelBridge, ParallelIterator};
use rayon::ThreadPoolBuilder;
use tf_idf_vectorizer::vectorizer::index::Index;


const MAX_SUDACHI_INPUT_SIZE: usize = 49100;

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
    let mut search_mode = true; // 検索モードを有効にする
    let index: Index<u16> = if fs::metadata("index.cbor").is_ok() {
        println!("index.cbor が見つかりました。以下のオプションから選択してください:");
        println!("1: 新しい index を作成する");
        println!("2: 既存の index に追加する");
        println!("3: 既存の index で検索のみ行う");
        let mut choice = String::new();
        io::stdin()
            .read_line(&mut choice)
            .expect("入力の読み込みに失敗しました");
        match choice.trim() {
            "1" => {
                println!("新しい index を作成します");
                search_mode = false; // ドキュメント追加モードに切り替え
                Index::new()
            }
            "2" | "3" => {
                let serialized = fs::read("index.cbor").expect("index.cbor の読み込みに失敗しました");
                let idx: Index<u16> =
                    serde_cbor::from_slice(&serialized).expect("index の復元に失敗しました");
                if choice.trim() == "2" {
                    println!("既存の index を読み込み、ドキュメントを追加します");
                    search_mode = false; // ドキュメント追加モードに切り替え
                } else {
                    println!("既存の index を読み込み、検索モードで実行します");
                }
                idx
            }
            _ => {
                println!("無効な選択です。新しい index を作成します");
                Index::new()
            }
        }
    } else {
        search_mode = false; // ドキュメント追加モードに切り替え
        Index::new()
    };

    let index = Arc::new(Mutex::new(index));
    if !search_mode {

        let dir_path = "Z:\\D\\dev\\web_dev\\idis_v2\\wikipedia_all_articles_fast";
        let blacklist = blacklist();

        // 2つのインデックス用の `DocumentAnalyzer` を作成
        // let analyzer2 = Arc::new(Mutex::new(DocumentAnalyzer::<String>::new()));
        let file_counter = Arc::new(AtomicUsize::new(0));
        let pool = ThreadPoolBuilder::new().num_threads(10).build().unwrap();
        pool.install(|| {
            // ファイルを逐次処理しつつ並列化
            fs::read_dir(dir_path)
                .expect("Failed to read directory")
                .par_bridge() // Rayonによる並列化
                .try_for_each(|entry| -> Result<(), ()> {
                    let entry = entry.expect("Failed to read entry");
                    let path = entry.path();

                    let file_count = file_counter.load(Ordering::Relaxed);

                    if file_count >= 100_000 { // すでに指定件数を超えている場合は処理をスキップ
                        return Err(()); 
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

                        // インデックスに追加
                        {
                            let mut idx = index.lock().expect("Failed to lock index");
                            idx.add_doc(format!("index1_{}", file_name), &token_refs);
                        }

                        // ファイルカウンターを増加
                        file_counter.fetch_add(1, Ordering::Relaxed);
                    }

                    Ok(())
                })
                .ok(); // エラーを無視して終了
        });
    }

    loop {
        println!("Enter your search query:");
        let mut query = String::new();
        io::stdin().read_line(&mut query).expect("Failed to read line");
        if query.trim() == "exit" {
            break; // "exit"と入力されたらループを終了
        }

        let query_tokens = tokenize_with_sudachi(&query, "B");
        let query_token_refs: Vec<&str> = query_tokens.iter().map(|s| s.as_str()).collect();
        println!("Query tokens: {:?}", query_token_refs);
        let query = {
            let idx = index.lock().expect("Failed to lock index");
            idx.generate_query(&query_token_refs)
        };
        let start = Instant::now();
        let result0 = {
            let idx = index.lock().expect("Failed to lock index");
            idx.search_cosine_similarity_parallel(&query, 16)
        };
        let duration = start.elapsed();
        let top_n = 10;
        println!("Search results (Top {} results, Time taken: {:.2?}):", top_n, duration);
        for (doc_id, similarity) in result0.into_iter().take(top_n) {
            println!("Document ID: {}, Similarity: {:.4}", doc_id, similarity);
        }
        println!("Search results (Time taken: {:.2?}):", duration);
    }
    {
        let idx = index.lock().expect("Failed to lock index");
        let serialized = serde_cbor::to_vec(&*idx).expect("Failed to serialize index");
        std::fs::write("index.cbor", serialized).expect("Failed to write index to file");
        println!("Serialized index has been saved as index.cbor");
    }
}

// fn main() {
//     // ここにメイン処理を記述
//     let mut index: Index<f32> = Index::new();

//     let doc1 = ["apple", "banana", "orange"];
//     let doc2 = ["banana", "grape", "kiwi"];
//     let doc3 = ["kiwi", "mango", "peach"];

//     index.add_doc("doc1".to_string(), &doc1);
//     index.add_doc("doc2".to_string(), &doc2);
//     index.add_doc("doc3".to_string(), &doc3);


//     let query = index.generate_query(&["apple", "banana", "orange"]);
//     let results = index.search_cosine_similarity(&query);

//     for (doc_id, similarity) in results {
//         println!("Document ID: {}, Similarity: {:.4}", doc_id, similarity);
//     }
// }