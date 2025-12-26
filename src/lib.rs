/// This crate is a Document Analysis Engine using TF-IDF Vectorizer.
pub mod vectorizer;
pub mod utils;


/// TF-IDF Vectorizer
/// 本クレートの最上位構造体であり、TF-IDFベクトライザーの主要な機能を提供します。
/// この構造体は、ドキュメント集合をTF-IDFベクトルに変換し、類似度計算や検索機能をサポートします。
/// 内部に以下のデータを保持します：
/// - コーパスのボキャブラリ
/// - 各ドキュメントのスパースTFベクトル
/// - TFベクトルの単語マップ
/// - IDFベクトルのキャッシュ
/// - TF-IDF計算エンジン
/// - ドキュメントの逆Index
/// 
/// `TFIDFVectorizer<N, K, E>`は以下のジェネリックパラメータを持ちます：
/// - `N`: ベクトルのパラメータ型(eg. f32, f64, u8, u16, u32)
/// - `K`: ドキュメントキーの型(eg. String, usize)
/// - `E`: TF-IDF計算エンジンの型(eg. DefaultTFIDFEngine)
/// 
/// インスタンス作成時に`Arc<Corpus>`としてコーパス参照を渡す必要があります。
/// `Corpus`は任意で差し替えたり、複数の`TFIDFVectorizer`インスタンス間で共有できます。
/// 
/// # Serialization
/// サポートしています。
/// このとき、`Corpus`参照も同封されます。
/// シリアライズ可能なデータ構造として`TFIDFData`を使用することもできます。
/// `TFIDFData`は`Corpus`参照を持たないため、`Corpus`と分離して保存したりできます。
/// 
/// # Deserialization
/// サポートされています。データの展開処理が含まれます。
pub use vectorizer::TFIDFVectorizer;

/// TF-IDF Vectorizer Data Structure for Serialization
/// この構造体は、`TFIDFVectorizer`で`Corpus`参照を持たないシリアライズ可能なデータ構造を提供します。
/// `into_tf_idf_vectorizer`メソッドを使用して`Arc<Corpus>`を渡し、`TFIDFVectorizer`に変換できます。
/// 
/// `TFIDFVectorizer`と比べて容量が小さくなります。
/// 
/// # Serialization
/// サポートされています。
/// 
/// # Deserialization
/// サポートされています。データの展開処理が含まれます。
pub use vectorizer::serde::TFIDFData;

/// Corpus for TF-IDF Vectorizer
/// この構造体はドキュメント集合を管理します。
/// ドキュメントの本文やIDは保持されず以下の情報のみを管理します：
/// - ドキュメント数
/// - コーパス全体での各トークンの出現ドキュメント数
/// 
/// IDF（逆文書頻度）計算の基礎データとして使用されます。
/// 
/// `TFIDFVectorizer`のインスタンス作成時に`Arc<Corpus>`として参照を渡す必要があります。
/// Corpus`はスレッドセーフであり、複数の`TFIDFVectorizer`インスタンス間で共有できます。
/// 
/// 統計、解析では`TokenFrequency`のほうが良いです。
/// 必要であれば`TokenFrequency`に変換できますが本質的に異なる意味をもつ統計データになることに注意してください。
/// 
/// # Thread Safety
/// この構造体はスレッドセーフであり、複数のスレッドから同時にアクセスできます。
/// DashMapとAtomicにより
pub use vectorizer::corpus::Corpus;

/// Token Frequency structure
/// ドキュメント内のトークン出現頻度を解析・管理するための構造体です。
/// 以下の情報を管理します：
/// - 各トークンの出現回数
/// - ドキュメント内の総トークン数
/// 
/// TF(Term Frequency)計算の基礎データとして使用されます。
/// 
/// トークンの追加、出現回数の設定、取得、統計情報の取得などの豊富な機能を提供します。
pub use vectorizer::token::TokenFrequency;

/// TF IDF Calculation Engine Trait
/// TF-IDF計算エンジンの振る舞いを定義するためのトレイトです。
/// 
/// このトレイトを実装することで、異なるTF-IDF計算戦略を`TFIDFVectorizer<E>`に組み込むことができます。
/// デフォルトの実装として`DefaultTFIDFEngine`が提供されており、教科書的なTF-IDF計算を行います。
/// 
/// デフォルトの実装では以下のパラメータ量子化が使用可能です:
/// - f16
/// - f32
/// - f64
/// - u8
/// - u16
/// - u32
pub use vectorizer::tfidf::{DefaultTFIDFEngine, TFIDFEngine};

/// Similarity Algorithm for TF-IDF Vectorizer
/// SimilarityAlgorithm列挙型は、TF-IDFベクトライザーで使用される類似度計算アルゴリズムを定義します。
/// 現在、以下のアルゴリズムがサポートされています：
/// - Contains: 単純包含チェック 単語を含むかどうかを確認します
/// - Dot: ドット積計算 長文検索に適しています
/// - Cosine Similarity: コサイン類似度計算 固有名詞の検索に適しています
/// - BM25 Like: BM25に類似したスコアリング 一般的な文書検索に適しています
pub use vectorizer::evaluate::scoring::SimilarityAlgorithm;

/// Query Structure for TF-IDF Vectorizer
/// この構造体は、TF-IDFベクトライザーで使用される検索クエリを表現します。
/// 複雑な論理条件を組み合わせてドキュメントをフィルタリングするための柔軟な手段を提供します。
pub use vectorizer::evaluate::query::Query;

/// Search Hits and Hit Entry structures
/// 検索結果を管理するためのデータ構造です。
/// - `Hits`: 検索結果のリストを保持し、スコアでソートする機能などを提供します。
/// - `HitEntry`: 各検索結果エントリを表し、ドキュメントキーとスコアを含みます。
pub use vectorizer::evaluate::scoring::{Hits, HitEntry};