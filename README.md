<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>Ultra-flexible & high-speed document analysis engine in Rust</p>
</div>

Supports everything from corpus construction → TF calculation → IDF calculation → TF-IDF vectorization / similarity search.

## Features
- Generic parameter engine (f32 / f64 / unsigned integer quantization)
- Full struct serialization / deserialization (`TFIDFData`) for persistence
- Similarity calculation utilities (`SimilarityQuery`, `Hits`) for search
- No index build step: instant add/remove, real-time operation
- Thread-safe
- Corpus info separated: index can be swapped independently
- Restorable: keeps document statistics

## Performance
- Environment
  - OS: Windows 11 Pro 23H2
  - CPU: i9-11900K @ 3.5GHz AVX512 enabled
  - RAM: 16GB + 32GB, DDR5 - 2800, 1-1
- Target
  - Wikipedia JP random 100k docs
- Results
  - tokenize = 0.00ms
  - build_refs = 0.00ms
  - tf_build = 0.00ms
  - score = 354.53ms
  - total = 354.53ms

## Setup
Cargo.toml
```toml
[dependencies]
tf-idf-vectorizer = "0.3.0"  # This README is for v0.3.x
```

## Basic Usage

```rust
use tf_idf_vectorizer::{Corpus, SimilarityQuery, TFIDFVectorizer, TokenFrequency};

fn main() {
    // Prepare corpus
    let corpus = Corpus::new();

    // Prepare token frequencies
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "fast", "parallel", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "flexible", "safe", "rust"]);

    // Create vectorizer (using u16 quantization)
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(&corpus);

    // Add documents to vectorizer
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);

    // Prepare query token frequency
    let mut query_tokens = TokenFrequency::new();
    query_tokens.add_tokens(&["rust", "fast"]);

    // Create query
    let query = SimilarityQuery::CosineSimilarity(query_tokens);
    // Calculate similarity and sort
    let mut result = vectorizer.similarity(query);
    result.sort_by_score();

    // Display results
    result.list.iter().for_each(|(k, s)| {
        println!("doc: {}, score: {}", k, s);
    });
}
```

## Serialization / Restoration
`TFIDFVectorizer` contains references and cannot be deserialized directly.  
Serialize as `TFIDFData`, and restore with `into_tf_idf_vectorizer(&Corpus)`.
You can use any corpus for restoration; if the index contains tokens not in the corpus, they are ignored.

```rust
// Save
let dump = serde_json::to_string(&vectorizer)?;

// Restore
let data: TFIDFData = serde_json::from_str(&dump)?;
let restored = data.into_tf_idf_vectorizer(&corpus);
```

## Similarity Search (Concept)
1. Convert input tokens to query vector (`SimilarityQuery`)
2. Compare with each document using dot product / cosine similarity, etc.
3. Return all results as `Hits`

You can inject your own scoring function by replacing the implemented `Compare` trait / `DefaultCompare`.

## Performance Tips
- Cache token dictionary (`token_dim_sample` / `token_dim_set`) to avoid rebuilding
- Sparse TF representation omits zeros
- Using integer scale types (u16/u32) compresses memory (normalization is just 1/max multiplication; float ops are slightly faster)
- Combine iterators to avoid temporary Vec allocation (`tf.zip(idf).map(...)`)

## Type Overview
| Type                | Role                                 |
|---------------------|--------------------------------------|
| Corpus              | Document set meta / frequency getter |
| TokenFrequency      | Token frequency in a single document |
| TFVector            | Sparse TF vector for one document    |
| IDFVector           | Global IDF and meta                  |
| TFIDFVectorizer     | TF/IDF management and search entry   |
| TFIDFData           | Intermediate for serialization       |
| DefaultTFIDFEngine  | TF/IDF calculation backend           |
| SimilarityQuery / Hits | Search query and results          |

## Customization
- Switch numeric type: f32/f64/u16/u32, etc.
- Extend scoring by implementing the `Compare` trait
- Swap out `TFIDFEngine` for different weighting schemes

## Examples (examples/)
Run the minimal example with:
```
cargo run --example basic
```

# Contributions welcome via Pull Request (。-`ω-