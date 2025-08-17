use std::{env, fs, io, path::Path, process::{Command, Stdio}, time::Instant};
use rayon::prelude::*;
use tf_idf_vectorizer::vectorizer::{corpus::Corpus, evaluate::scoring::SimilarityQuery, token::TokenFrequency, TFIDFVectorizer};

// Sudachi 側の "Input is too long" エラー (約49149 bytes) を避けるため余裕を持った上限
// 実際の CLI 実装内部バッファに安全マージンをとり 40KB とする
const SUDACHI_CHUNK_BYTE_LIMIT: usize = 40_000;

// 外部コマンド (Sudachi) を 1 回だけ実行して text をトークン化
fn sudachi_tokenize_once(cmd: &str, text: &str) -> io::Result<Vec<String>> {
    let mut child = Command::new(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(text.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr_s = String::from_utf8_lossy(&output.stderr);
        if stderr_s.contains("Input is too long") {
            // 呼び出し側で再分割戦略をとるためエラー返却
            return Err(io::Error::new(io::ErrorKind::Other, "sudachi input too long"));
        }
        eprintln!("[warn] sudachi command exited with status {:?}: {}", output.status.code(), stderr_s.trim());
        return Ok(Vec::new());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut tokens = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "EOS" { continue; }
        if let Some(tok) = line.split(|c: char| c == '\t' || c.is_whitespace()).next() {
            if tok == "EOS" { continue; }
            if !tok.is_empty() { tokens.push(tok.to_string()); }
        }
    }
    Ok(tokens)
}

// バイト上限を超える入力を UTF-8 文字境界で安全にチャンクへ分割
fn split_into_chunks(text: &str, limit: usize) -> Vec<String> {
    if text.as_bytes().len() <= limit { return vec![text.to_string()]; }
    let mut chunks = Vec::new();
    let mut buf = String::with_capacity(limit + 16);
    for line in text.split_inclusive('\n') { // 改行保持
        let mut rest = line;
        while !rest.is_empty() {
            let remaining = limit - buf.as_bytes().len();
            if remaining == 0 { // flush
                chunks.push(std::mem::take(&mut buf));
                continue;
            }
            if rest.as_bytes().len() <= remaining { // 全部入る
                buf.push_str(rest);
                rest = "";
            } else {
                // 部分だけ入れる。UTF-8 境界でカット
                // remaining バイト以内最大の char 境界位置を探す
                let mut cut = 0usize;
                for (idx, _) in rest.char_indices() { if idx <= remaining { cut = idx; } else { break; } }
                if cut == 0 { // 1 文字すら入らない -> flush してやり直し (バッファ空のはず)
                    chunks.push(std::mem::take(&mut buf));
                    continue;
                }
                buf.push_str(&rest[..cut]);
                rest = &rest[cut..];
            }
            if buf.as_bytes().len() >= limit { chunks.push(std::mem::take(&mut buf)); }
        }
    }
    if !buf.is_empty() { chunks.push(buf); }
    chunks
}

// 大きな入力をチャンクに分割して順次 Sudachi へ渡す (失敗した場合さらに細分化)
fn sudachi_tokenize(cmd: &str, text: &str) -> io::Result<Vec<String>> {
    // まず一次分割
    let mut chunks = split_into_chunks(text, SUDACHI_CHUNK_BYTE_LIMIT);
    let mut tokens_all = Vec::new();
    for chunk in chunks.drain(..) {
        match sudachi_tokenize_once(cmd, &chunk) {
            Ok(toks) => tokens_all.extend(toks),
            Err(e) => {
                // さらに半分に再分割して再試行 (再帰的)
                if e.to_string().contains("too long") && chunk.as_bytes().len() > 1024 {
                    let sub_limit = (chunk.as_bytes().len()/2).clamp(1024, SUDACHI_CHUNK_BYTE_LIMIT-1);
                    for sub in split_into_chunks(&chunk, sub_limit) {
                        let subtoks = sudachi_tokenize_once(cmd, &sub).unwrap_or_default();
                        tokens_all.extend(subtoks);
                    }
                } else {
                    eprintln!("[warn] sudachi tokenize chunk failed: {} (skipped {} bytes)", e, chunk.len());
                }
            }
        }
    }
    Ok(tokens_all)
}

fn detect_sudachi_cmd() -> String {
    if let Ok(cmd) = env::var("SUDACHI_CMD") { return cmd; }
    // 単純に既定名
    "sudachi".to_string()
}

// 旧: 事前スキャン + Rayon 並列 (総数とETAが出せる)
fn load_documents_parallel<P: AsRef<Path>>(dir: P, cmd: &str, _corpus: &Corpus, vectorizer: &mut TFIDFVectorizer<f32>, limit: Option<usize>) -> io::Result<usize> {
    use std::io::Write;
    let start_all = Instant::now();
    eprintln!("[stage] scanning directory ...");
    // 事前に対象ファイル一覧収集 (ディレクトリ内直下のみ)
    let mut files: Vec<_> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    let total = files.len();
    if total == 0 { eprintln!("[warn] no files found in directory"); return Ok(0); }
    let threads = rayon::current_num_threads();
    eprintln!("[stage] building TF in parallel (threads={})", threads);

    // 並列処理: 各ファイルを並列にトークン化 + TF 構築 (リアルタイム進捗: mpsc チャネルで逐次通知)
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let spin_frames = ["|","/","-","\\"]; // 簡易スピナー

    files.par_iter().for_each_with(tx.clone(), |tx, path| {
        let content = fs::read_to_string(path).unwrap_or_default();
        if content.trim().is_empty() { let _ = tx.send(None); return; }
        let tokens = match sudachi_tokenize(cmd, &content) {
            Ok(t) => t,
            Err(_) => { let _ = tx.send(None); return; }
        };
        if tokens.is_empty() { let _ = tx.send(None); return; }
        let token_len = tokens.len();
        let mut tf = TokenFrequency::new();
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        tf.add_tokens(&refs);
        let doc_key = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let file_size = fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
        let _ = tx.send(Some((doc_key, tf, token_len, file_size)));
    });
    drop(tx); // 送信終了

    let mut count = 0usize;          // 実際に add_doc 済み
    let mut processed = 0usize;      // 受信したメッセージ数 (スキップ含む)
    let mut total_tokens: usize = 0; 
    let mut total_bytes: u64 = 0;    
    let mut last_refresh = Instant::now();
    let mut spin_idx = 0usize;

    for msg in rx { // 並列完了順で飛んでくる
        processed += 1;
        if let Some((doc_key, tf, token_len, file_size)) = msg {
            vectorizer.add_doc(doc_key, &tf);
            count += 1;
            total_tokens += token_len;
            total_bytes += file_size;
            if let Some(lim) = limit { if count >= lim { break; } }
        }
        // 表示更新
        let pct = (processed as f64 / total as f64) * 100.0;
        let elapsed = start_all.elapsed();
        let elapsed_s = elapsed.as_secs_f64();
        let docs_per_sec = if elapsed_s > 0.0 { count as f64 / elapsed_s } else { 0.0 };
        let eta = if count > 0 { (total - processed) as f64 / docs_per_sec.max(1e-9) } else { f64::NAN };
        if last_refresh.elapsed().as_millis() >= 80 || processed == total {
            let frame = spin_frames[spin_idx % spin_frames.len()];
            spin_idx += 1;
            eprint!(
                "\r[indexing {}] {}/{} ({:.1}%) processed | added={} | tokens={} | bytes={} | {:.2} docs/s | ETA {:.1}s",
                frame,
                processed,
                total,
                pct,
                count,
                total_tokens,
                total_bytes,
                docs_per_sec,
                eta
            );
            let _ = io::stderr().flush();
            last_refresh = Instant::now();
        }
    }
    eprintln!();
    let total_elapsed = start_all.elapsed().as_secs_f64();
    eprintln!(
        "[done] indexed {} docs (processed {} files) | tokens={} | bytes={} | elapsed {:.2}s | avg {:.2} docs/s", 
        count, processed, total_tokens, total_bytes, total_elapsed, if total_elapsed>0.0 { count as f64 / total_elapsed } else { 0.0 }
    );
    Ok(count)
}

// (以前のパイプ一括読み取り関数は未使用のため削除)

// 新: デフォルト高速ストリーミング (ディレクトリ列挙しつつ即投入 / ワーカ並列 / 総件数未知)
fn load_documents_stream<P: AsRef<Path>>(dir: P, cmd: &str, _corpus: &Corpus, vectorizer: &mut TFIDFVectorizer<f32>, limit: Option<usize>) -> io::Result<usize> {
    use std::io::Write;
    use std::sync::mpsc;
    let start_all = Instant::now();
    let spin_frames = ["|","/","-","\\"];
    let mut spin_idx = 0usize;
    let workers = rayon::current_num_threads().max(2); // 少なくとも2

    // チャネル: パス送信用 & 結果受信用
    let (path_tx, path_rx_raw) = mpsc::channel::<std::path::PathBuf>();
    let path_rx = std::sync::Arc::new(std::sync::Mutex::new(path_rx_raw));
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (res_tx, res_rx) = mpsc::channel();

    // ワーカースレッド生成
    for _ in 0..workers {
        let res_tx = res_tx.clone();
        let cmd = cmd.to_string();
        let path_rx_cl = path_rx.clone();
        let stop_cl = stop_flag.clone();
        std::thread::spawn(move || {
            loop {
                if stop_cl.load(Ordering::Relaxed) { break; }
                let path = {
                    let lock = path_rx_cl.lock().unwrap();
                    lock.recv()
                };
                let Ok(path) = path else { break; };
                let file_size = fs::metadata(&path).ok().map(|m| m.len()).unwrap_or(0);
                let content = fs::read_to_string(&path).unwrap_or_default();
                if content.trim().is_empty() { let _ = res_tx.send(None); continue; }
                let tokens = match sudachi_tokenize(&cmd, &content) { Ok(t) => t, Err(_) => { let _ = res_tx.send(None); continue; } };
                if tokens.is_empty() { let _ = res_tx.send(None); continue; }
                let token_len = tokens.len();
                let mut tf = TokenFrequency::new();
                let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
                tf.add_tokens(&refs);
                let doc_key = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
                let _ = res_tx.send(Some((doc_key, tf, token_len, file_size)));
            }
        });
    }
    // res_tx はワーカーでクローンされているのでここで一旦落とす
    drop(res_tx);

    // 列挙スレッド (即座にワーカーへ流す)
    let dir_path_buf = dir.as_ref().to_path_buf();
    let enum_tx = path_tx.clone();
    let stop_enum = stop_flag.clone();
    let enum_handle = std::thread::spawn(move || {
        if let Ok(read_dir) = fs::read_dir(&dir_path_buf) {
            for e in read_dir.flatten() {
                let p = e.path();
                if p.is_file() {
                    if stop_enum.load(Ordering::Relaxed) { break; }
                    let _ = enum_tx.send(p);
                }
            }
        }
        // 送信終了
        // drop(enum_tx) here
    });
    drop(path_tx); // メイン側送信クローズ -> 列挙終了後にワーカーへEOFが流れる

    // 進捗集計
    let mut added_docs = 0usize; // vectorizer.add_doc 済み
    let mut processed_files = 0usize; // 成功/失敗含む受信件数
    let mut total_tokens = 0usize;
    let mut total_bytes: u64 = 0;
    let mut last_refresh = Instant::now();
    while let Ok(msg) = res_rx.recv() {
        processed_files += 1;
        if let Some((doc_key, tf, token_len, file_size)) = msg {
            vectorizer.add_doc(doc_key, &tf);
            added_docs += 1;
            total_tokens += token_len;
            total_bytes += file_size;
            if let Some(lim) = limit { if added_docs >= lim { stop_flag.store(true, Ordering::Relaxed); break; } }
        }
        if last_refresh.elapsed().as_millis() >= 120 {
            let elapsed = start_all.elapsed().as_secs_f64();
            let frame = spin_frames[spin_idx % spin_frames.len()];
            spin_idx += 1;
            let docs_per_sec = if elapsed > 0.0 { added_docs as f64 / elapsed } else { 0.0 };
            eprint!("\r[stream {}] added={} files={} tokens={} bytes={} {:.2} docs/s elapsed {:.1}s", frame, added_docs, processed_files, total_tokens, total_bytes, docs_per_sec, elapsed);
            let _ = io::stderr().flush();
            last_refresh = Instant::now();
        }
    }
    // 列挙終了待ち
    let _ = enum_handle.join();
    let elapsed = start_all.elapsed().as_secs_f64();
    let docs_per_sec = if elapsed > 0.0 { added_docs as f64 / elapsed } else { 0.0 };
    eprintln!("\r[done stream] added={} files={} tokens={} bytes={} elapsed {:.2}s avg {:.2} docs/s        ", added_docs, processed_files, total_tokens, total_bytes, elapsed, docs_per_sec);
    Ok(added_docs)
}

fn main() {
    let program_start = Instant::now();
    // ---- 簡易 CLI 引数処理 ----
    // --docs DIR       : 文書ディレクトリ (デフォ: data/ex_docs)
    // --sudachi CMD    : Sudachi コマンド (環境変数 SUDACHI_CMD も可)
    // --query "TEXT"   : クエリ文字列 (未指定なら stdin 全読み込みを試行)
    // 例)  echo "検索したい文章" | tf-idf-vectorizer
    //      tf-idf-vectorizer --query "検索したい文章" --docs ./data/ex_docs

    let mut args = env::args().skip(1); // program 名除外
    let mut docs_dir = String::from("C:\\Users\\minai\\Downloads\\wikipedia_all_articles_fast\\wikipedia_all_articles_fast");
    let mut sudachi_cmd_opt: Option<String> = None;
    let mut query_opt: Option<String> = None;
    let mut force_parallel = false; // --parallel で旧方式
    let mut limit_opt: Option<usize> = None; // --limit N
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
            "--stream" => { /* 既定でストリームなので何もしない */ }
            "--parallel" => { force_parallel = true; }
            "--limit" => {
                if let Some(v) = args.next() { match v.parse::<usize>() { Ok(n) if n>0 => limit_opt = Some(n), _ => { eprintln!("[error] --limit needs positive integer"); return; } } } else { eprintln!("[error] --limit requires a number"); return; }
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
    let mut vectorizer: TFIDFVectorizer<f32> = TFIDFVectorizer::new(&corpus);
    let load_start = Instant::now();
    let load_res = if force_parallel { load_documents_parallel(&docs_dir, &sudachi_cmd, &corpus, &mut vectorizer, limit_opt) } else { load_documents_stream(&docs_dir, &sudachi_cmd, &corpus, &mut vectorizer, limit_opt) };
    match load_res {
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

fn run_single_query(sudachi_cmd: &str, vectorizer: &mut TFIDFVectorizer<f32>, query_text: String) {
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

fn run_interactive(sudachi_cmd: &str, vectorizer: &mut TFIDFVectorizer<f32>) {
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