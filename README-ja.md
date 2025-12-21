<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>Rust製の 圧倒的柔軟/高速 な文章解析エンジン</p>
</div>

lang [ [en](./README.md) | ja ]
 
コーパス構築 → TF 計算 → IDF 計算 → TF-IDF ベクトル化 / 類似度検索 までを一通りサポート。

## 特徴
- ジェネリックパラメータ (f32 / f64 / 符号なし整数など) 対応エンジン 量子化
- 全構造体 シリアライズ / デシリアライズ (`TFIDFData`) 永続化
- 類似度計算ユーティリティ (`SimilarityAlgorithm`, `Hits`) 検索用途
- index構築処理なし 即時追加 削除 リアルタイム製
- スレッド間安全
- コーパス情報の分離 indexに対して差し替え可能
- 復元性 ドキュメントの統計情報を保持

## セットアップ
Cargo.toml
```toml
[dependencies]
tf-idf-vectorizer = "0.7"  # 本READMEは `v0.7.x` 向け
```

## 基本的な使い方

```rust
use std::sync::Arc;
use tf_idf_vectorizer::{Corpus, SimilarityQuery, TFIDFVectorizer, TokenFrequency};

fn main() {
    // コーパスを用意
    let corpus = Arc::new(Corpus::new());

    // token頻度の用意
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "高速", "並列", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "柔軟", "安全", "rust"]);

    // vectorizerの用意 内部パラメータはu16量子化に指定
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(corpus);    

    // vectorizerにドキュメントを挿入
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);e

    // query用のtoken頻度を用意
    let mut query_tokens = TokenFrequency::new();
    query_tokens.add_tokens(&["rust", "高速"]);
    let query = SimilarityQuery::CosineSimilarity(query_tokens);
    let mut result = vectorizer.similarity(query);
    result.sort_by_score();

    // 表示
    result.list.iter().for_each(|(k, s, l)| {
        println!("doc: {}, score: {}, length: {}", k, s, l);
    });    

    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer);
}
```

## シリアライズ / 復元
`TFIDFVectorizer` は参照を含むためデシリアライズ不可。  
シリアライズでは `TFIDFData` に強制され、復元時に `into_tf_idf_vectorizer(&Corpus)` で復元可能。
このときコーパスはセットのもの以外の任意のものでも正常に動作します(indexにcorpusない単語がある場合無視されます。)

```rust
// 保存
let dump = serde_json::to_string(&vectorizer)?;

// 復元
let data: TFIDFData = serde_json::from_str(&dump)?;
let restored = data.into_tf_idf_vectorizer(&corpus);
```

## 類似度検索 (概念)
1. 入力トークンをクエリベクトル化 (SimilarityAlgorithm)
2. 内積 / コサイン等で各ドキュメントと比較
3. すべての結果を Hits で返却

## パフォーマンス指針
- トークン辞書 (token_dim_sample / token_dim_set) は再構築を避けキャッシュ
- TF スパース化でゼロ省略
- 整数スケール型 (u16/u32) を使うとメモリ圧縮 (正規化時は 1/max 乗算のみ 演算はfloatのほうが少し高速です)
- 逆Indexを即時生成

## 型概要
| 型 | 役割 |
|----|------|
| Corpus | 文書集合メタ/頻度取得 |
| TokenFrequency | 単一文書内のトークン頻度 |
| TFVector | 1 文書の TF スパースベクトル |
| IDFVector | 全体 IDF とメタ |
| TFIDFVectorizer | TF/IDF 管理と検索入口 |
| TFIDFData | シリアライズ用中間体 |
| DefaultTFIDFEngine | TF/IDF 計算バックエンド |
| SimilarityAlgorithm / Hits | 検索クエリと結果 |

## カスタマイズ
- 数値型を f32/f64/u16/u32 などに切替
- TFIDFEngine を差し替えて異なる重み付け方式実験

## 例 (examples/)
`cargo run --example basic` で最小例を実行。  

# 貢献はプルリクで(。-`ω-)
