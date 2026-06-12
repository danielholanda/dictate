// Embeddable Lemonade (`lemond`) launcher.
//
// Runs the bundled `lemond` binary as a private localhost subprocess so that the
// "local" transcription provider can hit an OpenAI-compatible
// `/api/v1/audio/transcriptions` endpoint backed by the NPU (recipe `flm`,
// model `whisper-v3-turbo-FLM`). The process is spawned with a fresh random API
// key and a free port per app launch, and is killed on app exit.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Manager};

/// Lemonade model served on the NPU for local transcription.
pub const LOCAL_MODEL: &str = "whisper-v3-turbo-FLM";

/// Live handle to the running `lemond` subprocess.
pub struct LemondHandle {
    pub port: u16,
    pub key: String,
    pub child: Child,
}

/// Tauri-managed state holding the running `lemond` instance (if any).
#[derive(Default)]
pub struct LemondState {
    pub inner: Mutex<Option<LemondHandle>>,
}

impl LemondState {
    /// Returns (port, key) if `lemond` is running, for use by the local provider.
    pub fn endpoint(&self) -> Option<(u16, String)> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|h| (h.port, h.key.clone())))
    }
}

fn bin_name() -> &'static str {
    if cfg!(windows) {
        "lemond.exe"
    } else {
        "lemond"
    }
}

/// Strip a Windows verbatim path prefix (`\\?\` or `\\?\UNC\`). Tauri's
/// `resource_dir()` returns verbatim paths, but lemond runs `flm list --json`
/// via `cmd /c`, and `cmd.exe` cannot operate with a `\\?\` working directory —
/// it fails with "The system cannot find the path specified", which silently
/// breaks NPU model discovery. Hand lemond a plain path instead.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        p
    }
}

/// Resolve the vendored `vendor/lemonade` directory. Prefers the bundled
/// resource dir; falls back to the source tree for `tauri dev`.
fn lemonade_dir(app: &AppHandle) -> PathBuf {
    if let Ok(res) = app.path().resource_dir() {
        let p = strip_verbatim(res.join("vendor").join("lemonade"));
        if p.join(bin_name()).exists() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vendor")
        .join("lemonade")
}

/// Pick a free localhost TCP port by binding to port 0 and reading it back.
fn free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Generate a 64-char hex token for the per-launch API key. The endpoint is
/// loopback-only with broadcast disabled, so this only needs to be unguessable
/// per run, not cryptographically perfect — derived from high-res time, pid and
/// stack-address entropy without pulling in an extra crate.
fn gen_key() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut seed = std::process::id() as u64;
    let mut out = String::with_capacity(64);
    for i in 0u64..4 {
        let mut h = DefaultHasher::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        now.as_nanos().hash(&mut h);
        seed.hash(&mut h);
        i.hash(&mut h);
        (&seed as *const u64 as usize).hash(&mut h);
        let v = h.finish();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(v ^ 0x9E37_79B9_7F4A_7C15);
        out.push_str(&format!("{:016x}", v));
    }
    out
}

/// Spawn the `lemond` subprocess. Returns the child plus its key and port.
fn spawn_process(app: &AppHandle) -> Result<(Child, String, u16), String> {
    let dir = lemonade_dir(app);
    let bin = dir.join(bin_name());
    if !bin.exists() {
        return Err(format!("lemond binary not found at {}", bin.display()));
    }

    let port = free_port().map_err(|e| format!("no free port: {e}"))?;
    let key = gen_key();

    // Capture lemond's own stdout/stderr to a log file so startup/model-load
    // failures are diagnosable (otherwise they vanish into the void).
    let log_path = std::env::temp_dir().join("dictate-lemond.log");
    let log = std::fs::File::create(&log_path).map_err(|e| format!("lemond log: {e}"))?;
    let log_err = log.try_clone().map_err(|e| format!("lemond log clone: {e}"))?;
    println!("[lemond] dir={} port={} log={}", dir.display(), port, log_path.display());

    // The directory doubles as lemond's cache_dir: it holds config.json and the
    // installed `bin/flm` backend. `--port` overrides whatever is in config.json.
    let mut cmd = Command::new(&bin);
    cmd.arg(&dir)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .env("LEMONADE_API_KEY", &key)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));

    // Suppress the console window that a GUI app would otherwise flash when
    // spawning the `lemond` console executable.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = cmd.spawn().map_err(|e| format!("failed to spawn lemond: {e}"))?;

    Ok((child, key, port))
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build lemond http client")
}

/// Poll `GET /api/v1/health` until it returns 200 or the timeout elapses.
async fn wait_for_health(port: u16, key: &str, timeout: Duration) -> Result<(), String> {
    let client = http_client();
    let url = format!("http://127.0.0.1:{port}/api/v1/health");
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Ok(resp) = client
            .get(&url)
            .header("Authorization", format!("Bearer {key}"))
            .send()
            .await
        {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err("lemond did not become healthy in time".to_string());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Ask lemond to load the model into memory (NPU) to avoid first-request cold
/// start. Best-effort: attempts a backend install + model pull on failure.
async fn preload_model(port: u16, key: &str) {
    let client = http_client();
    let base = format!("http://127.0.0.1:{port}/api/v1");
    let load = || async {
        client
            .post(format!("{base}/load"))
            .header("Authorization", format!("Bearer {key}"))
            .json(&serde_json::json!({ "model_name": LOCAL_MODEL }))
            .send()
            .await
    };

    match load().await {
        Ok(resp) if resp.status().is_success() => {
            println!("[lemond] preloaded {LOCAL_MODEL} on NPU");
            return;
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eprintln!("[lemond] initial load failed ({status}): {body}; attempting recovery");
            // Backend missing -> install; model missing -> pull. Both idempotent.
            let _ = client
                .post(format!("{base}/install"))
                .header("Authorization", format!("Bearer {key}"))
                .json(&serde_json::json!({ "recipe": "flm", "backend": "npu" }))
                .send()
                .await;
            let _ = client
                .post(format!("{base}/pull"))
                .header("Authorization", format!("Bearer {key}"))
                .json(&serde_json::json!({ "model_name": LOCAL_MODEL }))
                .send()
                .await;
        }
        Err(e) => {
            eprintln!("[lemond] load request error: {e}");
            return;
        }
    }

    // Retry once after recovery.
    match load().await {
        Ok(resp) if resp.status().is_success() => {
            println!("[lemond] preloaded {LOCAL_MODEL} after recovery");
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            eprintln!("[lemond] model preload still failing ({status}): {body}");
        }
        Err(e) => eprintln!("[lemond] model preload retry error: {e}"),
    }
}

/// Start `lemond`, store its handle in `LemondState`, wait for health and
/// preload the model. Intended to run in a background task at app startup.
pub async fn launch(app: AppHandle) {
    let (child, key, port) = match spawn_process(&app) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[lemond] failed to start: {e}");
            return;
        }
    };
    println!("[lemond] started on 127.0.0.1:{port}");

    {
        let state = app.state::<LemondState>();
        let mut guard = state.inner.lock().expect("LemondState lock poisoned");
        *guard = Some(LemondHandle {
            port,
            key: key.clone(),
            child,
        });
    }

    if let Err(e) = wait_for_health(port, &key, Duration::from_secs(90)).await {
        eprintln!("[lemond] {e}");
        return;
    }
    println!("[lemond] healthy");

    preload_model(port, &key).await;
}

/// Kill the `lemond` subprocess if running. Safe to call multiple times.
///
/// lemond spawns its own `flm.exe` NPU server as a child; a plain `kill()` of
/// `lemond` would orphan that grandchild, leaving it holding the (single) NPU
/// and ~1.2GB of RAM. On Windows we therefore kill the whole process tree via
/// `taskkill /T`; elsewhere we fall back to killing the direct child.
pub fn shutdown(app: &AppHandle) {
    if let Some(state) = app.try_state::<LemondState>() {
        if let Ok(mut guard) = state.inner.lock() {
            if let Some(mut handle) = guard.take() {
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                    let _ = Command::new("taskkill")
                        .args(["/PID", &handle.child.id().to_string(), "/T", "/F"])
                        .creation_flags(CREATE_NO_WINDOW)
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                }
                let _ = handle.child.kill();
                let _ = handle.child.wait();
                println!("[lemond] stopped");
            }
        }
    }
}
