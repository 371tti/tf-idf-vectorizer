use std::collections::HashMap;
use sprs::CsVec;
use tf_idf_vectorizer::token::DocumentAnalyzer;

// あなたの提示したコードファイルと同一ディレクトリに置く、またはモジュールとして使う場合は適宜 `mod` 宣言してください。
// ここではシンプルにすべて同じファイルにある想定で書いています。
//
// use crate::DocumentAnalyzer;   // モジュールの場合の例
// use crate::Index;             // モジュールの場合の例

fn main() {
    // 1. DocumentAnalyzerを初期化
    let mut analyzer = DocumentAnalyzer::<String>::new();

    // 2. ドキュメントを追加
    //   - 第1引数: ID (ここではString)
    //   - 第2引数: ドキュメントのトークン配列（&strのスライス）
    //   - 第3引数: テキストの一部や本文 (Option<&str>)
    // サンプルとして3つのドキュメントを用意
    let text1 = "Rust is fast and memory-safe.";
    let text2 = "Rust is a cool systems programming language. It is memory-safe and blazingly fast. Rust is also quite popular. It is a modern language. Rust is quite a nice language.";
    let text3 = "Java is quite verbose but popular.";

    // テキストをwhitespaceでトークン分割
    let tokens1: Vec<&str> = text1.split_whitespace().collect();
    let tokens2: Vec<&str> = text2.split_whitespace().collect();
    let tokens3: Vec<&str> = text3.split_whitespace().collect();

    // DocumentAnalyzerを生成
    let mut analyzer = DocumentAnalyzer::<String>::new();

    // doc1を追加
    analyzer.add_document(
        "doc1".to_string(),
        &tokens1,
        Some(text1),  // テキストをそのまま保持
    );

    // doc2を追加
    analyzer.add_document(
        "doc2".to_string(),
        &tokens2,
        Some(text2),
    );

    // doc3を追加
    analyzer.add_document(
        "doc3".to_string(),
        &tokens3,
        Some(text3),
    );

    // 追加結果を確認
    println!("Total documents in analyzer: {}", analyzer.get_document_count());
    for (doc_id, doc) in &analyzer.documents {
        println!("ID: {}, Tokens: {:?}", doc_id, doc.tokens.token_count);
    }
    println!();

    // 3. インデックスを生成
    let index = analyzer.generate_index();
    println!("{:?}", index.get_index());

    // 4. 検索する
    //   - 引数1: クエリトークンのスライス
    //   - 引数2: 上位何件を取得するか(n)
    let query_tokens = &tokens2;
    let n = 10;
    let search_results = index.search(query_tokens, n);
    println!("Search results (top {}):", n);

    if search_results.is_empty() {
        println!("  No similar documents found.");
    } else {
        for (doc_id, similarity) in &search_results {
            println!("  Document ID: {}, similarity={:.4}", doc_id, similarity);
        }
    }

    // 5. 適宜、ドキュメントの削除や更新なども可能
    //   - 例: analyzer.del_document("doc3") など

    // 最後に区切り
    println!("\nDone.");
}
