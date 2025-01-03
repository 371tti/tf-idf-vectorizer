このリポジトリは、idis-rustのサブモジュールです。

Rustで実装されたtf-idfベクトライザーです。
tf-idf（Term Frequency-Inverse Document Frequency）は、情報検索やテキストマイニングで使用される重要な技術で、文書内の単語の重要度を評価します。

## 使い方
1. Cargo.tomlに本クレートを追加
2. DocumentAnalyzerを用いて文書追加・インデックス生成
3. 検索メソッドでクエリ実行

その他SplitterやToken頻度解析、正規化、ソートなどを提供します。

## 使い方の詳細

以下に、`tf-idf-vectorizer` クレートの基本的な使用方法の例を示します。

## 主な特徴
- 高速なTF-IDFベクトル化
- BM25などの多様な検索オプションを提供

## インデックス合成例
- 複数の DocumentAnalyzer から生成した Index を synthesize_index で統合し、一括検索が可能
- 同一のキーがある場合、後に渡した Index により上書きされる

## BM25活用例
- 検索時に BM25パラメータ(k1, b)を指定し、より高品質なランキングを実現