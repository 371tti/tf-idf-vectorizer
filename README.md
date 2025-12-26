<div align="center">
<h1 style="font-size: 50px">TF‑IDF-Vectorizer</h1>
<p>An overwhelmingly flexible and fast text analysis engine written in Rust</p>
</div>

lang [ [en](./README.md) | ja ]
 
Supports the full pipeline: corpus building → TF calculation → IDF calculation → TF‑IDF vectorization / similarity search.

## Features
- Engine with generic parameters (f32 / f64 / unsigned integers, etc.) and quantization support
- Serde support
- Similarity computation utilities (`SimilarityAlgorithm`, `Hits`, `Query`) for search use cases
- No index build step; supports immediate add/delete in real time
- Thread-safe
- Corpus information is separable and can be swapped for an index
- Restorable: preserves document statistics

## Setup
Cargo.toml
```toml
[dependencies]
tf-idf-vectorizer = "0.9"  # This README targets `v0.9.x`
```

## Basic usage

```rust
use std::sync::Arc;

use tf_idf_vectorizer::{Corpus, SimilarityAlgorithm, TFIDFVectorizer, TokenFrequency, vectorizer::evaluate::query::Query};

fn main() {
    // build corpus
    let corpus = Arc::new(Corpus::new());

    // make token frequencies
    let mut freq1 = TokenFrequency::new();
    freq1.add_tokens(&["rust", "fast", "parallel", "rust"]);
    let mut freq2 = TokenFrequency::new();
    freq2.add_tokens(&["rust", "flexible", "safe", "rust"]);

    // add documents to vectorizer
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(corpus);    
    vectorizer.add_doc("doc1".to_string(), &freq1);
    vectorizer.add_doc("doc2".to_string(), &freq2);
    vectorizer.del_doc(&"doc1".to_string());
    vectorizer.add_doc("doc3".to_string(), &freq1);

    let query = Query::and(Query::token("rust"), Query::token("safe"));
    let algorithm = SimilarityAlgorithm::CosineSimilarity;
    let mut result = vectorizer.search(&algorithm, query);
    result.sort_by_score_desc();

    // print result
    println!("Search Results: \n{}", result);
    // debug
    println!("result count: {}", result.list.len());
    println!("{:?}", vectorizer);
}
```

## Serialization / Restore
Because `TFIDFVectorizer` contains references, it cannot be deserialized directly.  
When serializing, it is converted into `TFIDFData`, and can be restored with `into_tf_idf_vectorizer(&Corpus)`.
At that time, any corpus (not only the original one) can be used and it will still work correctly (tokens that exist in the index but not in the corpus are ignored).

```rust
// save
let dump = serde_json::to_string(&vectorizer)?;

// restore
let data: TFIDFData = serde_json::from_str(&dump)?;
let restored = data.into_tf_idf_vectorizer(&corpus);
```

## Performance guidelines
- Cache token dictionaries (token_dim_sample / token_dim_set) to avoid rebuilding
- Sparsify TF vectors to omit zeros
- Using integer-scaled types (u16/u32) reduces memory usage (during normalization, only 1/max multiplication is needed; floats are slightly faster for computation)
- Build the inverted index on the fly

## Type overview
| Type | Role |
|----|------|
| Corpus | Document set metadata / frequency lookup |
| TokenFrequency | Token frequencies within a single document |
| TFVector | TF sparse vector for one document |
| IDFVector | Global IDF and metadata |
| TFIDFVectorizer | TF/IDF management and search entry point |
| TFIDFData | Intermediate type for serialization |
| DefaultTFIDFEngine | Backend for TF/IDF computation |
| SimilarityAlgorithm / Hits / Query | Search query and results |

## Customization
- Switch numeric type to f16/f32/f64/u16/u32, etc.
- Replace `TFIDFEngine` to use different weighting schemes

## Examples (examples/)
Run the minimal example with `cargo run --example basic`.  

# Contributions via pull requests, please (。-`ω-)
