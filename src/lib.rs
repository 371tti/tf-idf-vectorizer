/// This crate is a Document Analysis Engine using a TF-IDF Vectorizer.
pub mod vectorizer;
pub mod utils;

/// TF-IDF Vectorizer
/// The top-level struct of this crate, providing the main TF-IDF vectorizer features.
/// It converts a document collection into TF-IDF vectors and supports similarity
/// computation and search functionality.
///
/// Internally, it holds:
/// - The corpus vocabulary
/// - Sparse TF vectors for each document
/// - A token map for TF vectors
/// - An IDF vector cache
/// - A TF-IDF calculation engine
/// - An inverted index of documents
///
/// `TFIDFVectorizer<N, K, E>` has the following generic parameters:
/// - `N`: Vector parameter type (e.g., f32, f64, u8, u16, u32)
/// - `K`: Document key type (e.g., String, usize)
/// - `E`: TF-IDF calculation engine type (e.g., DefaultTFIDFEngine)
///
/// When creating an instance, you must pass a corpus reference as `Arc<Corpus>`.
/// The `Corpus` can optionally be replaced, and can be shared among multiple
/// `TFIDFVectorizer` instances.
///
/// # Serialization
/// Supported.
/// In this case, the `Corpus` reference is included as well.
/// You can also use `TFIDFData` as a serializable data structure.
/// `TFIDFData` does not hold a `Corpus` reference, so it can be stored separately
/// from the `Corpus`.
///
/// # Deserialization
/// Supported, including data expansion/unpacking.
pub use vectorizer::TFIDFVectorizer;

/// TF-IDF Vectorizer Data Structure for Serialization
/// This struct provides a serializable data structure that does not hold a `Corpus`
/// reference (unlike `TFIDFVectorizer`).
/// You can convert it into `TFIDFVectorizer` by passing an `Arc<Corpus>` via
/// `into_tf_idf_vectorizer`.
///
/// Compared to `TFIDFVectorizer`, it has a smaller footprint.
///
/// # Serialization
/// Supported.
///
/// # Deserialization
/// Supported, including data expansion/unpacking.
pub use vectorizer::serde::TFIDFData;

/// Corpus for TF-IDF Vectorizer
/// This struct manages a collection of documents.
/// It does not store document text or IDs; it only manages:
/// - The number of documents
/// - The number of documents in which each token appears across the corpus
///
/// It is used as the base data for IDF (Inverse Document Frequency) calculation.
///
/// When creating a `TFIDFVectorizer`, you must pass a corpus reference as
/// `Arc<Corpus>`.
/// `Corpus` is thread-safe and can be shared among multiple `TFIDFVectorizer`
/// instances.
///
/// For statistics/analysis, `TokenFrequency` may be more suitable.
/// You can convert to `TokenFrequency` if needed, but note that it represents
/// fundamentally different statistical meaning.
///
/// # Thread Safety
/// This struct is thread-safe and can be accessed concurrently from multiple threads.
/// Implemented using DashMap and atomics.
pub use vectorizer::corpus::Corpus;

/// Token Frequency structure
/// A struct for analyzing/managing token occurrence frequency within a document.
/// It manages:
/// - The count of occurrences of each token
/// - The total number of tokens in the document
///
/// Used as base data for TF (Term Frequency) calculation.
///
/// Provides rich functionality such as adding tokens, setting/getting counts,
/// and retrieving statistics.
pub use vectorizer::token::TokenFrequency;

/// TF IDF Calculation Engine Trait
/// A trait that defines the behavior of a TF-IDF calculation engine.
///
/// By implementing this trait, you can plug different TF-IDF calculation strategies
/// into `TFIDFVectorizer<E>`.
/// A default implementation, `DefaultTFIDFEngine`, is provided and performs
/// textbook-style TF-IDF calculation.
///
/// The default implementation supports the following parameter quantizations:
/// - f16
/// - f32
/// - f64
/// - u8
/// - u16
/// - u32
pub use vectorizer::tfidf::{DefaultTFIDFEngine, TFIDFEngine};

/// Similarity Algorithm for TF-IDF Vectorizer
/// The `SimilarityAlgorithm` enum defines similarity-scoring algorithms used by the
/// TF-IDF vectorizer.
///
/// Currently, the following algorithms are supported:
/// - Contains: simple containment check (whether it contains the token)
/// - Dot: dot product (suitable for long-document search)
/// - Cosine Similarity: cosine similarity (suitable for proper noun search)
/// - BM25 Like: BM25-like scoring (suitable for general document search)
pub use vectorizer::evaluate::scoring::SimilarityAlgorithm;

/// Query Structure for TF-IDF Vectorizer
/// Represents a search query used by the TF-IDF vectorizer.
/// It provides a flexible way to filter documents by combining complex logical
/// conditions.
pub use vectorizer::evaluate::query::Query;

/// Search Hits and Hit Entry structures
/// Data structures for managing search results.
/// - `Hits`: holds a list of search results and provides features such as sorting by score
/// - `HitEntry`: represents a single result entry, containing the document key and score
pub use vectorizer::evaluate::scoring::{Hits, HitEntry};