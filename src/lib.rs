//! # TF-IDF Vectorizer
//!
//! This crate provides a **document analysis engine** based on a highly customizable
//! **TF-IDF vectorizer**.
//!
//! It is designed for:
//! - Full-text search engines
//! - Document similarity analysis
//! - Large-scale corpus processing
//!
//! ## Architecture Overview
//!
//! The crate is composed of the following core concepts:
//!
//! - **Corpus**: Global document-frequency statistics (IDF base)
//! - **TermFrequency**: Per-document term statistics (TF base)
//! - **TFIDFVectorizer**: Converts documents into sparse TF-IDF vectors
//! - **TFIDFEngine**: Pluggable TF / IDF calculation strategy
//! - **SimilarityAlgorithm**: Multiple scoring algorithms (Cosine, Dot, BM25-like)
//!
//! ## Example
//!
//! ```rust
//! use std::sync::Arc;
//! 
//! use half::f16;
//! use tf_idf_vectorizer::{Corpus, SimilarityAlgorithm, TFIDFVectorizer, TermFrequency, vectorizer::evaluate::query::Query};
//! 
//! fn main() {
//!     // build corpus
//!     let corpus = Arc::new(Corpus::new());
//! 
//!     // make term frequencies
//!     let mut freq1 = TermFrequency::new();
//!     freq1.add_terms(&["rust", "高速", "並列", "rust"]);
//!     let mut freq2 = TermFrequency::new();
//!     freq2.add_terms(&["rust", "柔軟", "安全", "rust"]);
//! 
//!     // add documents to vectorizer
//!     let mut vectorizer: TFIDFVectorizer<f16> = TFIDFVectorizer::new(corpus);    
//!     vectorizer.add_doc("doc1".to_string(), &freq1);
//!     vectorizer.add_doc("doc2".to_string(), &freq2);
//!     vectorizer.del_doc(&"doc1".to_string());
//!     vectorizer.add_doc("doc3".to_string(), &freq1);
//! 
//!     let query = Query::and(Query::term("rust"), Query::term("安全"));
//!     let algorithm = SimilarityAlgorithm::CosineSimilarity;
//!     let mut result = vectorizer.search(&algorithm, query);
//!     result.sort_by_score_desc();
//! 
//!     // print result
//!     println!("Search Results: \n{}", result);
//!     // debug
//!     println!("result count: {}", result.list.len());
//!     println!("{:?}", vectorizer);
//! }
//! ```
//!
//! ## Thread Safety
//!
//! - `Corpus` is thread-safe and can be shared across vectorizers
//! - Designed for parallel indexing and search workloads
//!
//! ## Serialization
//!
//! - `TFIDFVectorizer` and `TFIDFData` support serialization
//! - `TFIDFData` does **not** hold a `Corpus` reference and is suitable for storage

pub mod vectorizer;
pub mod utils;

#[doc = "## Core Vectorizer"]
/// TF-IDF Vectorizer
///
/// The top-level struct of this crate, providing the main TF-IDF vectorizer features.
///
/// It converts a document collection into TF-IDF vectors and supports similarity
/// computation and search functionality.
///
/// ### Internals
/// - Corpus vocabulary
/// - Sparse TF vectors per document
/// - term index mapping
/// - Cached IDF vector
/// - Pluggable TF-IDF engine
/// - Inverted document index
///
/// ### Type Parameters
/// - `N`: Vector parameter type (e.g., `f32`, `f64`, `u16`)
/// - `K`: Document key type (e.g., `String`, `usize`)
/// - `E`: TF-IDF calculation engine
///
/// ### Notes
/// - Requires an `Arc<Corpus>` on construction
/// - `Corpus` can be shared across multiple vectorizers
///
/// ### Serialization
/// Supported.  
/// Serialized data includes the `Corpus` reference.
///
/// For corpus-independent storage, use [`TFIDFData`].
pub use vectorizer::TFIDFVectorizer;

#[doc = "## Serializable Data Structures"]
/// TF-IDF Vectorizer Data Structure (Corpus-free)
///
/// A compact, serializable representation of a TF-IDF vectorizer.
///
/// Unlike [`TFIDFVectorizer`], this struct does **not** hold a `Corpus` reference.
/// It can be converted back into a `TFIDFVectorizer` by providing an `Arc<Corpus>`.
///
/// ### Use Cases
/// - Persistent storage
/// - Network transfer
/// - Memory-efficient snapshots
///
/// ### Serialization
/// Supported.
///
/// ### Deserialization
/// Supported, including internal data expansion.
pub use vectorizer::serde::TFIDFData;

#[doc = "## Corpus & Statistics"]
/// Corpus for TF-IDF Vectorizer
///
/// Manages global document-frequency statistics required for IDF calculation.
///
/// This struct does **not** store document text or identifiers.
/// It only tracks:
/// - Total number of documents
/// - Number of documents containing each term
///
/// ### Thread Safety
/// - Fully thread-safe
/// - Implemented using `DashMap` and atomics
///
/// ### Notes
/// - Must be shared via `Arc<Corpus>`
/// - Can be reused across multiple vectorizers
pub use vectorizer::corpus::Corpus;

/// term Frequency Structure
///
/// Manages per-document term statistics used for TF calculation.
///
/// Tracks:
/// - term occurrence counts
/// - Total term count in the document
///
/// ### Use Cases
/// - TF calculation
/// - term-level statistics
pub use vectorizer::term::TermFrequency;

#[doc = "## TF-IDF Engines"]
/// TF-IDF Calculation Engine Trait
///
/// Defines the behavior of a TF-IDF calculation engine.
///
/// Custom engines can be implemented and plugged into
/// [`TFIDFVectorizer`].
///
/// A default implementation, [`DefaultTFIDFEngine`], is provided.
///
/// ### Supported Numeric Types
/// - `f16`
/// - `f32`
/// - `f64`
/// - `u8`
/// - `u16`
/// - `u32`
pub use vectorizer::tfidf::{DefaultTFIDFEngine, TFIDFEngine};

#[doc = "## Similarity & Search"]
/// Similarity Algorithm
///
/// Defines scoring algorithms used during search.
///
/// ### Variants
/// - `Contains`: term containment check
/// - `Dot`: Dot product (long documents)
/// - `Cosine`: Cosine similarity (proper nouns)
/// - `BM25Like`: BM25-inspired scoring
pub use vectorizer::evaluate::scoring::SimilarityAlgorithm;

/// Query Structure
///
/// Represents a search query with logical filtering conditions.
pub use vectorizer::evaluate::query::Query;

/// Search Results
///
/// - `Hits`: A collection of ranked search results
/// - `HitEntry`: A single search result entry
pub use vectorizer::evaluate::scoring::{Hits, HitEntry};
