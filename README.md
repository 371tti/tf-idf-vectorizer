<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>A Rust-based, extremely flexible and high-performance text analysis engine</p>
</div>

lang [ en | [jp](./README-ja.md)  ]
 
Supports the full pipeline: corpus construction → TF calculation → IDF calculation → TF‑IDF vectorization / similarity search.

## Features
- Engine with generic parameters (f32 / f64 / unsigned integers, etc.) and quantization support
- All core structs are serializable / deserializable (`TFIDFData`) for persistence
- Similarity utilities (`SimilarityAlgorithm`, `Hits`) for search use cases
- No index-building step — immediate add/remove; real-time updates
- Thread-safe
- Corpus information separated and replaceable with respect to the index
- Restorability — retains document statistics

## Setup
Cargo.toml
```toml
[dependencies]
tf-idf-vectorizer = "0.7"  # This README is targeted at `v0.7.x`
```

## Basic usage

```rust
use std::sync::Arc;
use tf_idf_vectorizer::{Corpus, SimilarityQuery, TFIDFVectorizer, TokenFrequency};

fn main() {
    // Prepare a corpus
    let corpus = Arc::new(Corpus::new());

    // Prepare token frequencies
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "fast", "parallel", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "flexible", "safe", "rust"]);

    // Create a vectorizer; internal type parameter set to u16 for quantization
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(corpus);    

    // Insert documents into the vectorizer
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);

    // Prepare query token frequencies
    let mut query_tokens = TokenFrequency::new();
    query_tokens.add_tokens(&["rust", "fast"]);
    let query = SimilarityQuery::CosineSimilarity(query_tokens);
    let mut result = vectorizer.similarity(query);
    result.sort_by_score();

    // Display
    result.list.iter().for_each(|(k, s, l)| {
        println!("doc: {}, score: {}, length: {}", k, s, l);
    });    

    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer);
}
```

## Serialization / Restoration
`TFIDFVectorizer` contains references and cannot be deserialized directly.  
For serialization it is converted to `TFIDFData`, and on restoration you can call `into_tf_idf_vectorizer(&Corpus)` to restore it.  
The corpus provided at restoration can be any corpus; terms not present in the corpus index will be ignored.

```rust
// Save
let dump = serde_json::to_string(&vectorizer)?;

// Restore
let data: TFIDFData = serde_json::from_str(&dump)?;
let restored = data.into_tf_idf_vectorizer(&corpus);
```

## Similarity search (concept)
1. Vectorize input tokens into a query vector (SimilarityAlgorithm)  
2. Compare with each document using dot product / cosine, etc.  
3. Return all results as Hits

## Performance guidelines
- Cache token dictionaries (token_dim_sample / token_dim_set) to avoid reconstruction
- Sparse TF to omit zeros
- Using integer scaled types (u16/u32) reduces memory (normalization uses 1/max multiplication; floating-point arithmetic is slightly faster)
- Generate reverse index immediately

## Type overview
| Type | Role |
|----|------|
| Corpus | Document collection metadata / frequency lookup |
| TokenFrequency | Token frequency within a single document |
| TFVector | TF sparse vector for a single document |
| IDFVector | Global IDF and metadata |
| TFIDFVectorizer | TF/IDF management and search entry point |
| TFIDFData | Intermediate form for serialization |
| DefaultTFIDFEngine | TF/IDF computation backend |
| SimilarityAlgorithm / Hits | Search query and results |

## Customization
- Switch numeric types (f32/f64/u16/u32, etc.)
- Replace TFIDFEngine to experiment with different weighting schemes

## Examples (examples/)
Run the minimal example with `cargo run --example basic`.  

Contributions via pull requests.
