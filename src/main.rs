mod token;
use token::TokenFrequency;

fn main() {
    // TokenFrequencyのインスタンスを作成
    let mut tf = TokenFrequency::new();

    let doc = "hello world im a rustacean. I love rust programming language. 
    I also like python and java. I have been programming in rust for 2 years now. 
    I have also worked with java for 3 years. I have a good understanding of c++ as well.
    I have been programming in rust for 2 years now. I have have have also worked with java for 3 years. I have a good understanding of c++ as well.";

    for word in doc.split_whitespace() {
        tf.add_token(word);
    }
    println!("Before removing stop tokens:");
    println!("Token count: {:?}", tf.get_token_count_hashmap());
    println!("Total token count: {}", tf.get_total_token_count());
    println!("Token list: {:?}", tf.get_tf_hashmap_parallel());

    // 削除対象のストップトークン
    let stop_words = ["rust", "java"];

    // 並列削除
    tf.remove_stop_tokens_parallel(&stop_words);

    println!("\nAfter removing stop tokens:");
    println!("Token count: {:?}", tf.get_token_count_hashmap());
    println!("Total token count: {}", tf.get_total_token_count());
}
