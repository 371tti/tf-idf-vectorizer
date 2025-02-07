use std::{path::PathBuf, sync::Arc};

use rayon::result;
use sudachi::{analysis::{stateless_tokenizer::StatelessTokenizer, Tokenize}, config::Config, dic::dictionary::JapaneseDictionary, prelude::{Mode, MorphemeList}};




pub struct JPTokenizer {
    tok: StatelessTokenizer<Arc<JapaneseDictionary>>,
}

impl JPTokenizer {
    pub fn new(config_path: &str) -> Self {
        let config = Config::new(Some(PathBuf::from(config_path)),None, None).expect("Failed to read config file");
        let dict = JapaneseDictionary::from_cfg(&config).expect("Failed to load dictionary");
        let tok = StatelessTokenizer::new(Arc::new(dict));
        Self { tok }
    }

    pub fn tokenize<'a>(
        &'a self,
        text: &'a str,
        mode: Mode,
    ) -> MorphemeList<Arc<JapaneseDictionary>> {
        let result = self.tok.tokenize(&text, mode, false);
        result.expect("Failed to tokenize")
    }

    pub fn dict(&self) -> &JapaneseDictionary {
        self.tok.as_dict()
    }
}