use std::{env, fs, io, path::Path, process::{Command, Stdio}, time::Instant};
use tf_idf_vectorizer::vectorizer::{corpus::Corpus, evaluate::scoring::SimilarityQuery, token::TokenFrequency, TFIDFVectorizer};

// 外部コマンド (Sudachi 系) を使って与えられたテキストをトークン列へ。
// 期待フォーマット: 1 行 1 トークン もしくは 形態素解析結果 (最初の列を表層形とみなす)
fn sudachi_tokenize(cmd: &str, text: &str) -> io::Result<Vec<String>> {
    // Sudachi は入力全体を一括で stdin へ渡す。
    let mut child = Command::new(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    // 書き込み
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(text.as_bytes())?;
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        eprintln!("[warn] sudachi command exited with status {:?}", output.status.code());
        return Ok(Vec::new());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut tokens = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "EOS" { continue; } // Sudachi / MeCab style end marker
        // タブ or 空白 split の先頭列を利用
        if let Some(tok) = line.split(|c: char| c == '\t' || c.is_whitespace()).next() {
            if tok == "EOS" { continue; }
            if !tok.is_empty() { tokens.push(tok.to_string()); }
        }
    }
    Ok(tokens)
}

fn detect_sudachi_cmd() -> String {
    if let Ok(cmd) = env::var("SUDACHI_CMD") { return cmd; }
    // 単純に既定名
    "sudachi".to_string()
}

fn load_documents<P: AsRef<Path>>(dir: P, cmd: &str, _corpus: &Corpus, vectorizer: &mut TFIDFVectorizer<u16>) -> io::Result<usize> {
    let mut count = 0usize;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() { continue; }
        let content = fs::read_to_string(&path).unwrap_or_default();
        if content.trim().is_empty() { continue; }
        let tokens = match sudachi_tokenize(cmd, &content) {
            Ok(t) => t,
            Err(e) => { eprintln!("[warn] tokenize failed for {:?}: {}", path, e); continue; }
        };
        if tokens.is_empty() { continue; }
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&token_refs);
        let doc_key = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        vectorizer.add_doc(doc_key, &tf);
        count += 1;
    }
    Ok(count)
}

// (以前のパイプ一括読み取り関数は未使用のため削除)

fn main() {
    let program_start = Instant::now();
    // ---- 簡易 CLI 引数処理 ----
    // --docs DIR       : 文書ディレクトリ (デフォ: data/ex_docs)
    // --sudachi CMD    : Sudachi コマンド (環境変数 SUDACHI_CMD も可)
    // --query "TEXT"   : クエリ文字列 (未指定なら stdin 全読み込みを試行)
    // 例)  echo "検索したい文章" | tf-idf-vectorizer
    //      tf-idf-vectorizer --query "検索したい文章" --docs ./data/ex_docs

    let mut args = env::args().skip(1); // program 名除外
    let mut docs_dir = String::from("data/ex_docs");
    let mut sudachi_cmd_opt: Option<String> = None;
    let mut query_opt: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--docs" => {
                if let Some(v) = args.next() { docs_dir = v; } else { eprintln!("[error] --docs requires a path"); return; }
            }
            "--sudachi" => {
                if let Some(v) = args.next() { sudachi_cmd_opt = Some(v); } else { eprintln!("[error] --sudachi requires a command name"); return; }
            }
            "--query" => {
                if let Some(v) = args.next() { query_opt = Some(v); } else { eprintln!("[error] --query requires a string"); return; }
            }
            "-h" | "--help" => {
                print_usage();
                return;
            }
            other => {
                // 位置引数をクエリとして解釈 (最初のみ)
                if query_opt.is_none() { query_opt = Some(other.to_string()); } else { eprintln!("[warn] extra arg ignored: {}", other); }
            }
        }
    }

    let sudachi_cmd = sudachi_cmd_opt.unwrap_or_else(|| detect_sudachi_cmd());

    // ---- 文書ロード ----
    let corpus = Corpus::new();
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(&corpus);
    let load_start = Instant::now();
    match load_documents(&docs_dir, &sudachi_cmd, &corpus, &mut vectorizer) {
        Ok(n) => eprintln!("[info] loaded {} documents from {}", n, docs_dir),
        Err(e) => { eprintln!("[error] failed to load documents: {}", e); return; }
    }
    if vectorizer.documents.is_empty() {
        eprintln!("[error] no documents loaded. abort");
        return;
    }
    let idf_start = Instant::now();
    vectorizer.update_idf();
    let indexing_done = Instant::now();
    eprintln!("[time] load_docs={:.2}ms update_idf={:.2}ms total_index={:.2}ms", 
        idf_start.duration_since(load_start).as_secs_f64()*1000.0,
        indexing_done.duration_since(idf_start).as_secs_f64()*1000.0,
        indexing_done.duration_since(load_start).as_secs_f64()*1000.0);

    // ---- モード判定: --query 指定時はその1回だけ、未指定なら対話ループ ----
    if let Some(qtext) = query_opt {
        run_single_query(&sudachi_cmd, &mut vectorizer, qtext);
    } else {
        run_interactive(&sudachi_cmd, &mut vectorizer);
    }

    eprintln!("[time] program_total={:.2}ms", program_start.elapsed().as_secs_f64()*1000.0);
}

fn print_usage() {
    eprintln!("Usage: tf-idf-vectorizer [--docs DIR] [--sudachi CMD] [--query \"TEXT\"]");
    eprintln!("If --query omitted, stdin is read. Output format: <score>\t<doc_key>");
}

fn run_single_query(sudachi_cmd: &str, vectorizer: &mut TFIDFVectorizer<u16>, query_text: String) {
    let q = query_text.trim();
    if q.is_empty() { eprintln!("[error] empty query"); return; }
    let t0 = Instant::now();
    let tokens = sudachi_tokenize(sudachi_cmd, q).unwrap_or_default();
    let tokens: Vec<String> = tokens.into_iter().filter(|t| t != "EOS").collect();
    let t1 = Instant::now();
    if tokens.is_empty() { eprintln!("[warn] no tokens"); return; }
    let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let t2 = Instant::now();
    let mut tf = TokenFrequency::new();
    tf.add_tokens(&refs);
    let t3 = Instant::now();
    let query = SimilarityQuery::CosineSimilarity(tf);
    let mut results = vectorizer.similarity(query);
    results.sort_by_score();
    let t4 = Instant::now();
    eprintln!("[query] tokens: {}", tokens.join(" "));
    eprintln!("[time] tokenize={:.2}ms build_refs={:.2}ms tf_build={:.2}ms score={:.2}ms total={:.2}ms", 
        t1.duration_since(t0).as_secs_f64()*1000.0,
        t2.duration_since(t1).as_secs_f64()*1000.0,
        t3.duration_since(t2).as_secs_f64()*1000.0,
        t4.duration_since(t3).as_secs_f64()*1000.0,
        t4.duration_since(t0).as_secs_f64()*1000.0);
    for (key, score) in results.list.iter() { println!("{}\t{}", score, key); }
}

fn run_interactive(sudachi_cmd: &str, vectorizer: &mut TFIDFVectorizer<u16>) {
    use std::io::{self, Write};
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("Query> ");
        let _ = stdout.flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() { eprintln!("[error] read error"); break; }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            eprintln!("[info] bye");
            break;
        }
        let tokens = sudachi_tokenize(sudachi_cmd, trimmed).unwrap_or_default();
    let tokens: Vec<String> = tokens.into_iter().filter(|t| t != "EOS").collect();
    let start = Instant::now(); // timing baseline for this query
        let t1 = Instant::now();
        if tokens.is_empty() { println!("(no tokens)" ); continue; }
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let t2 = Instant::now();
        let mut tf = TokenFrequency::new();
        tf.add_tokens(&refs);
        let t3 = Instant::now();
        let query = SimilarityQuery::CosineSimilarity(tf);
        let mut results = vectorizer.similarity(query);
        results.sort_by_score();
        let t4 = Instant::now();
        eprintln!("[query] tokens: {}", tokens.join(" "));
        eprintln!("[time] tokenize={:.2}ms build_refs={:.2}ms tf_build={:.2}ms score={:.2}ms total={:.2}ms", 
            t1.duration_since(start).as_secs_f64()*1000.0,
            t2.duration_since(t1).as_secs_f64()*1000.0,
            t3.duration_since(t2).as_secs_f64()*1000.0,
            t4.duration_since(t3).as_secs_f64()*1000.0,
            t4.duration_since(start).as_secs_f64()*1000.0);
        for (key, score) in results.list.iter().take(20) { // 上位20件
            println!("{score}\t{key}");
        }
    }
}