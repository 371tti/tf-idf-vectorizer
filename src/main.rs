use std::{env, fs, io, path::Path, process::{Command, Stdio}, time::Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use tf_idf_vectorizer::vectorizer::{corpus::Corpus, evaluate::scoring::SimilarityQuery, token::TokenFrequency, TFIDFVectorizer, serde::TFIDFData};
use serde::{Serialize, de::DeserializeOwned};

// Sudachi 側の "Input is too long" エラー (約49149 bytes) を避けるため余裕を持った上限
// 実際の CLI 実装内部バッファに安全マージンをとり 40KB とする
const SUDACHI_CHUNK_BYTE_LIMIT: usize = 40_000;

// Sudachi が存在しない場合のフォールバックフラグ (一度失敗したら以降 spawn を試さない)
static SUDACHI_FALLBACK: AtomicBool = AtomicBool::new(false);

// 外部コマンド (Sudachi) を 1 回だけ実行して text をトークン化
fn sudachi_tokenize_once(cmd: &str, text: &str) -> io::Result<Vec<String>> {
    if SUDACHI_FALLBACK.load(Ordering::Relaxed) {
        return Ok(text.split_whitespace().map(|s| s.to_string()).collect());
    }
    // Sudachi コマンド起動。存在しない/起動失敗時はフォールバックして空白区切りトークン化
    let mut child = match Command::new(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[warn] failed to spawn '{}': {} -> fallback whitespace tokenization (cached)", cmd, e);
            SUDACHI_FALLBACK.store(true, Ordering::Relaxed);
            return Ok(text.split_whitespace().map(|s| s.to_string()).collect());
        }
    };
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
fn load_documents_parallel<P: AsRef<Path>>(dir: P, cmd: &str, _corpus: &Corpus, vectorizer: &mut TFIDFVectorizer<u16>, limit: Option<usize>) -> io::Result<usize> {
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
    // bounded channel to provide backpressure and avoid unbounded memory growth
    let (tx, rx) = mpsc::sync_channel(threads.saturating_mul(4));
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
    // reduce internal capacity before sending to lower peak memory
    tf.shrink_to_fit();
    // free large temporaries (content, tokens) as soon as possible to reduce peak memory
    drop(tokens);
    drop(content);
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
fn load_documents_stream<P: AsRef<Path>>(dir: P, cmd: &str, _corpus: &Corpus, vectorizer: &mut TFIDFVectorizer<u16>, limit: Option<usize>) -> io::Result<usize> {
    use std::io::Write;
    use std::sync::mpsc;
    let start_all = Instant::now();
    let spin_frames = ["|","/","-","\\"];
    let mut spin_idx = 0usize;
    let workers = rayon::current_num_threads().max(2); // 少なくとも2

    // まず軽量にファイル一覧だけ収集して総数を把握 (サイズ取得等は後で)
    // 大量ファイル時に無音にならないよう進捗表示
    eprint!("[scan] collecting file list ...");
    let mut file_list: Vec<std::path::PathBuf> = Vec::new();
    let mut scan_count: usize = 0;
    let mut collected = 0usize;
    for e in fs::read_dir(&dir)? {
        if let Ok(e) = e {
            let p = e.path();
            if p.is_file() {
                file_list.push(p);
                collected += 1;
                if let Some(lim) = limit { if collected >= lim { scan_count += 1; break; } }
            }
            scan_count += 1;
            if scan_count % 1000 == 0 { eprint!("\r[scan] visited={} files (current kept={})", scan_count, file_list.len()); let _ = io::stderr().flush(); }
        }
        if let Some(lim) = limit { if collected >= lim { break; } }
    }
    eprintln!("\r[scan] visited={} files (kept={}) done", scan_count, file_list.len());
    let total_files_all = file_list.len();
    if total_files_all == 0 { eprintln!("[warn] no files found in directory"); return Ok(0); }
    eprintln!("[scan] total_files={} starting workers (limit={:?})", total_files_all, limit);
    let target_total = limit.map(|l| l.min(total_files_all)).unwrap_or(total_files_all);

    // チャネル: パス送信用 & 結果受信用
    // bounded channels give backpressure when workers or enumerator are producing faster than consuming
    let (path_tx, path_rx_raw) = mpsc::sync_channel::<std::path::PathBuf>(workers.saturating_mul(4));
    let path_rx = std::sync::Arc::new(std::sync::Mutex::new(path_rx_raw));
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (res_tx, res_rx) = mpsc::sync_channel(workers.saturating_mul(4));

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
                // reduce internal capacity before sending to lower peak memory
                tf.shrink_to_fit();
                // drop large temporaries early to lower peak memory
                drop(tokens);
                drop(content);
                let doc_key = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
                let _ = res_tx.send(Some((doc_key, tf, token_len, file_size)));
            }
        });
    }
    // res_tx はワーカーでクローンされているのでここで一旦落とす
    drop(res_tx);

    // 列挙スレッド (即座にワーカーへ流す)
    let enum_tx = path_tx.clone();
    let stop_enum = stop_flag.clone();
    let enum_handle = std::thread::spawn(move || {
        let mut sent = 0usize;
        for p in file_list.into_iter() {
            if stop_enum.load(Ordering::Relaxed) { break; }
            if let Some(lim) = limit {
                if sent >= lim { break; }
                let _ = enum_tx.send(p.clone()); sent += 1;
            } else {
                let _ = enum_tx.send(p); sent += 1;
            }
        }
        // 送信終了 (drop)
    });
    drop(path_tx); // メイン側送信クローズ -> 列挙終了後にワーカーへEOFが流れる

    // 進捗集計
    let mut added_docs = 0usize; // vectorizer.add_doc 済み
    let mut processed_files = 0usize; // 成功/失敗含む受信件数
    let mut total_tokens = 0usize;
    let mut total_bytes: u64 = 0;
    let mut last_refresh = Instant::now();
    let mut ewma_rate: f64 = 0.0; // 平滑化した docs/s
    let alpha = 0.2; // EWMA 係数
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
            let inst_rate = if elapsed > 0.0 { added_docs as f64 / elapsed } else { 0.0 };
            ewma_rate = if ewma_rate == 0.0 { inst_rate } else { ewma_rate * (1.0 - alpha) + inst_rate * alpha };
            let remaining = if added_docs >= target_total { 0 } else { target_total - added_docs };
            let eta = if ewma_rate > 0.0 { remaining as f64 / ewma_rate } else { f64::NAN };
            eprint!("\r[stream {}] {}/{} added | files_processed={} tokens={} bytes={} rate={:.2} docs/s ETA={:.1}s elapsed {:.1}s", frame, added_docs, target_total, processed_files, total_tokens, total_bytes, ewma_rate, eta, elapsed);
            let _ = io::stderr().flush();
            last_refresh = Instant::now();
        }
    }
    // 列挙終了待ち
    let _ = enum_handle.join();
    let elapsed = start_all.elapsed().as_secs_f64();
    let docs_per_sec = if elapsed > 0.0 { added_docs as f64 / elapsed } else { 0.0 };
    eprintln!("\r[done stream] added={} target={} files_processed={} tokens={} bytes={} elapsed {:.2}s avg {:.2} docs/s        ", added_docs, target_total, processed_files, total_tokens, total_bytes, elapsed, docs_per_sec);
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
    let mut save_base: Option<String> = None; // --save name -> name.index.cbor + name.corpus.cbor
    let mut load_base: Option<String> = None; // --load name -> name.index.cbor + name.corpus.cbor
    let mut skip_build: bool = false; // --load-index のみで検索 (コーパスは空で良い場合)
    // true if we successfully loaded both corpus and index from --load
    let mut loaded = false;
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
            "--save" => {
                if let Some(v) = args.next() { save_base = Some(v); } else { eprintln!("[error] --save requires a name"); return; }
            }
            "--load" => {
                if let Some(v) = args.next() { load_base = Some(v); } else { eprintln!("[error] --load requires a name"); return; }
            }
            "--no-indexing" => { skip_build = true; }
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

    // ---- コーパス & インデックス構築 / 読み込み ----
    // create empty corpus first and optionally overwrite from load
    let mut corpus: Corpus = Corpus::new();
    // 量子化型 u16 採用
    let mut vectorizer: TFIDFVectorizer<u16> = TFIDFVectorizer::new(&corpus);

    let load_start = Instant::now();
    if let Some(ref base) = load_base {
        let index_path = format!("{}.index.cbor", base);
        let corpus_path = format!("{}.corpus.cbor", base);
        // only treat as loaded if both files successfully read
        let corpus_res = load_cbor::<Corpus>(&corpus_path);
        let index_res = load_cbor::<TFIDFData<u16, String>>(&index_path);
        match (corpus_res, index_res) {
            (Ok(c), Ok(data)) => {
                // assign outer corpus so it lives long enough
                corpus = c;
                vectorizer = data.into_tf_idf_vectorizer(&corpus);
                eprintln!("[info] loaded corpus+index from {} (docs={})", base, vectorizer.documents.len());
                loaded = true;
            }
            (Ok(_), Err(e)) => {
                eprintln!("[warn] failed to load index {}: {} -> will try full build", index_path, e);
                if skip_build { return; }
            }
            (Err(e), Ok(_)) => {
                eprintln!("[warn] loaded index but failed to load corpus {}: {} -> will try full build", corpus_path, e);
                if skip_build { return; }
            }
            (Err(e1), Err(e2)) => {
                eprintln!("[warn] failed to load corpus/index {}: {}, {} -> will try full build", base, e1, e2);
                if skip_build { return; }
            }
        }
    }
    if !loaded && !skip_build {
        let load_res = if force_parallel { load_documents_parallel(&docs_dir, &sudachi_cmd, &corpus, &mut vectorizer, limit_opt) } else { load_documents_stream(&docs_dir, &sudachi_cmd, &corpus, &mut vectorizer, limit_opt) };
        match load_res {
            Ok(n) => eprintln!("[info] loaded {} documents (vocab={}) from {}", n, corpus.vocab_size(), docs_dir),
            Err(e) => { eprintln!("[error] failed to load documents: {}", e); return; }
        }
    if vectorizer.documents.is_empty() { eprintln!("[error] no documents loaded. abort"); return; }
    vectorizer.update_idf();
        if let Some(ref base) = save_base {
            let index_path = format!("{}.index.cbor", base);
            let corpus_path = format!("{}.corpus.cbor", base);
            if let Err(e) = save_cbor(&index_path, &vectorizer) { eprintln!("[warn] save index failed: {}", e); } else { eprintln!("[info] index saved to {}", index_path); }
            if let Err(e) = save_cbor(&corpus_path, &corpus) { eprintln!("[warn] save corpus failed: {}", e); } else { eprintln!("[info] corpus saved to {}", corpus_path); }
        }
    }
    let indexing_done = Instant::now();
    eprintln!("[time] index_total={:.2}ms", indexing_done.duration_since(load_start).as_secs_f64()*1000.0);

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
    let query = SimilarityQuery::BM25(tf, 1.5, 0.75); // BM25 example
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
    for (key, score, _) in results.list.iter() { println!("{}\t{}", score, key); }
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
        let query = SimilarityQuery::BM25(tf, 1.5, 0.75); // BM25 example
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
        for (key, score, doc_len) in results.list.iter().take(20) { // 上位20件
            println!("{score}\t\t{doc_len}\t\t{key}");
        }
    }
}

// ---- 汎用: CBOR 保存/読込 ----
fn save_cbor<T: Serialize>(path: &str, value: &T) -> io::Result<()> {
    use std::io::Write;
    let tmp = format!("{}.tmp", path);
    let mut file = fs::File::create(&tmp)?;
    serde_cbor::to_writer(&mut file, value).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    file.flush()?;
    #[cfg(unix)]
    { use std::os::unix::fs::FileExt; file.sync_all()?; }
    #[cfg(windows)]
    { file.sync_all()?; }
    fs::rename(tmp, path)?;
    Ok(())
}

fn load_cbor<T: DeserializeOwned>(path: &str) -> io::Result<T> {
    let file = fs::File::open(path)?;
    match serde_cbor::from_reader(file) {
        Ok(v) => Ok(v),
        Err(e) => {
            let meta = fs::metadata(path).ok();
            if let Some(m) = meta { eprintln!("[warn] load_cbor failed {}, file_size={}", e, m.len()); }
            return Err(io::Error::new(io::ErrorKind::Other, e));
        }
    }
}