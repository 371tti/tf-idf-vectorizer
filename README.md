# tf-idf-Vectorizer

開発中です
破壊的変更をする予定です

このライブラリは、Rustで実装されたTF-IDFベクトライザーです。  
TF-IDF (Term Frequency-Inverse Document Frequency) は、文書中の各単語の重要度を評価するための有力な手法で、情報検索やテキストマイニング、自然言語処理の分野で広く利用されています。  
本ライブラリは、高速な並列処理と効率的なアルゴリズムにより、大規模なコーパスに対しても実用的なパフォーマンスを提供します。
  
- **TF (Term Frequency)**: 各ドキュメント内で単語がどれだけ登場するか  
- **IDF (Inverse Document Frequency)**: コーパス全体における単語の希少性を評価  
これらを組み合わせることにより、他の単語に比べてその単語がどれだけ特徴的かを示すことができます。  
本ライブラリはこの計算を効率的に行い、検索や解析タスクへの応用を容易にします。

## 概要
ライブラリは以下のプロセスでTF-IDFを算出します：
1. **トークン抽出と前処理**:  
   - 各ドキュメントから有意な単語（トークン）を抽出  
   - 小文字化、フィルタリング、正規化などの前処理を実施
2. **TF計算**:  
   - 各ドキュメント内での各トークンの出現回数を記録  
   - 正規化して文書内での相対頻度を算出
3. **IDF計算**:  
   - コーパス全体での各トークンの出現する文書数から逆文書頻度を計算  
4. **TF-IDFベクトル生成**:  
   - 各トークンについてTFとIDFを掛け合わせ、重要度を評価  
   - 結果は得点順にソートされる

内部実装では `TokenFrequency` によりこれらの計算が管理され、Rayon ライブラリを用いた並列処理で高速かつスケーラブルに動作します。

## 特徴
- **高速並列処理**: 大量のデータを効率的に処理  
- **柔軟な前処理**: トークンのフィルタリングや正規化が容易にカスタマイズ可能  
- **シンプルなAPI**: インデックス作成、ドキュメント追加、検索をシンプルなメソッドで実現

## インストール
Cargo.tomlに以下の依存関係を追加してください:
```toml
[dependencies]
tf-idf-vectorizer = "0.1.0"
```

## 使い方

### インデックス作成とドキュメント追加
下記は、インデックス作成とドキュメントの追加、検索クエリの生成例です。
```rust
use tf_idf_vectorizer::vectorizer::index::Index;

fn main() {
    // インデックスを初期化
    let mut index = Index::new();
    
    // 例: フルーツに関するドキュメントを追加
    index.add_doc("doc1".to_string(), &["apple", "banana", "orange"]);
    
    // 検索クエリの生成例（"apple" と "banana" を含む文書を検索）
    let query = index.generate_query(&["apple", "banana"]);
    let results = index.search_cosine_similarity_parallel(&query, 16);
    println!("検索結果: {:?}", results);
}
```

### TokenFrequency の利用例
`TokenFrequency` は、各ドキュメント内の単語の出現情報を管理し、TF/IDFの計算を行います。
```rust
use tf_idf_vectorizer::vectorizer::token::TokenFrequency;

fn main() {
    // TokenFrequency の初期化
    let mut tf = TokenFrequency::new();
    
    // ドキュメント内の単語を登録
    tf.add_tokens(&["apple", "banana", "apple", "orange"]);
    
    // "apple" の出現頻度 (TF) を取得
    println!("apple の頻度: {}", tf.tf_token("apple"));
    
    // 仮に全ドキュメント数を10としてIDFを計算
    for (token, idf) in tf.idf_vector_ref_str(10) {
        println!("Token: {}, IDF: {:.4}", token, idf);
    }
}
```
### TFIDFVectorizer の利用例

以下のコード例は、TFIDFVectorizer の基本的な利用方法を示しています。  
各文書からコーパスを構築し、特定の単語リストに対して TF-IDF ベクトルを算出する手順を確認できます。

```rust
use tf_idf_vectorizer::vectorizer::tfidf::TFIDFVectorizer;

fn main() {
    // TFIDFVectorizer の新規作成
    let mut vectorizer = TFIDFVectorizer::new();
    
    // サンプルのコーパス（各文書は単語のスライス）
    let documents = vec![
        vec!["rust", "高速", "パフォーマンス", "並列処理"],
        vec!["tf-idf", "ベクトライザー", "テキストマイニング"],
        vec!["rust", "tf-idf", "アルゴリズム", "効率的"],
    ];
    
    // 各文書からトークンを追加してコーパスを構築
    for doc in &documents {
        vectorizer.add_corpus(doc);
    }
    
    // 特定の単語リストに対してTF-IDFベクトルを算出
    let tokens = vec!["rust", "tf-idf", "並列処理"];
    let tfidf_vector = vectorizer.tf_idf_vector(&tokens);
    
    println!("TF-IDF Vector: {:?}", tfidf_vector);
}
```
###

## API リファレンス

### TFIDFVectorizer
- `TFIDFVectorizer::new()`  
  新規のベクトライザーを生成します。
  
- `add_corpus(tokens: &[&str])`  
  渡されたトークンをコーパスに追加し、内部の頻度データを更新します。
  
- `tf_idf_vector(tokens: &[&str]) -> Vec<(&str, f64)>`  
  指定されたトークンリストからTF-IDFベクトルを算出し、得点順にソートして返します。

### TokenFrequency
- `TokenFrequency::new()`  
  新しい TokenFrequency のインスタンスを生成します。
  
- `add_tokens(tokens: &[&str])`  
  ドキュメント内のトークンを登録し、出現カウントを更新します。
  
- `tf_token(token: &str) -> f64`  
  指定したトークンの正規化された出現頻度 (TF) を計算して返します。
  
- `idf_vector_ref_str(total_doc_count: u64) -> Vec<(&str, f64)>`  
  コーパス全体に基づき、各トークンの逆文書頻度 (IDF) を算出して返します。

## 内部実装の詳細
主要なコンポーネントは以下の通りです：
- **インデックス管理 (`Index`)**:  
  - ドキュメントの追加や検索クエリの生成を行い、類似性計算を担当。  
- **頻度管理 (`TokenFrequency`)**:  
  - 個々の単語の出現情報を格納し、TFとIDFの計算ロジックを提供。  
- **並列処理**:  
  - Rayonを利用して、複数のドキュメントの処理・検索を並列化し、パフォーマンスを向上。  

各コンポーネントはモジュールごとに分離されており、拡張や機能のカスタマイズも容易に行えます。

## ライセンス
このプロジェクトは MIT ライセンスの下で提供されています。

