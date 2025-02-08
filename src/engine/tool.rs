

// use crate::vectorizer::{analyzer::DocumentAnalyzer, index::Index};

// use super::jp_tokenizer::tokenizer::{self, JPTokenizer};

// pub enum TokenizeMode {
//     Lite,
//     Balanced,
//     Speed,
//     Quality,
// }
// pub struct SearchSystem {
//     pub indexes: Vec<Index<String>>,
//     pub analyzer: DocumentAnalyzer<String>,
//     pub tokenizer: JPTokenizer,
// }

// impl SearchSystem {
//     pub fn new(sudachi_dic_config_path: &str) -> Self {
//         let analyzer = DocumentAnalyzer::new();
//         let indexes = Vec::new();
//         let tokenizer = JPTokenizer::new(sudachi_dic_config_path);
//         Self { indexes, analyzer , tokenizer}
//     }

//     pub fn add_document(&mut self, id: String, content: &str, tokenize_mode: TokenizeMode,text: Option<&str>) {
//         match tokenize_mode {
//             TokenizeMode::Lite => {
//                 let ms = self.tokenizer.tokenize(content, sudachi::prelude::Mode::B);
//                 ms.iter().collect();
//             }
//             TokenizeMode::Balanced => {
//                 self.tokenizer.tokenize(content, sudachi::prelude::Mode::A);
//             }
//             TokenizeMode::Speed => {
//                 self.tokenizer.tokenize(content, tokenizer::Mode::C);
//             }
//             TokenizeMode::Quality => {
//                 self.tokenizer.tokenize(content, tokenizer::Mode::D);
//             }
            
//         }
//         self.tokenizer.tokenize(content, )
//         self.analyzer.add_document(id, content, text);
//     }

//     pub fn build_index(&mut self) {
//         let index = self.analyzer.generate_index();
//         self.indexes.push(index);
//     }


// }