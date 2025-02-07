

use crate::vectorizer::{analyzer::DocumentAnalyzer, index::Index};

pub struct SearchSystem {
    pub indexes: Vec<Index<String>>,
    pub analyzer: DocumentAnalyzer<String>,
}

impl SearchSystem {
    pub fn new() -> Self {
        let analyzer = DocumentAnalyzer::new();
        let indexes = Vec::new();
        Self { indexes, analyzer }
    }
}