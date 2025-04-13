use token::TokenFrequency;

pub mod index;
pub mod token;
pub mod analyzer;

pub struct TFIDFVectorizer {
    pub token_counter: TokenFrequency,
}