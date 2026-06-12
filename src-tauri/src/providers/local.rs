// Local (on-device, NPU) transcription provider.
//
// Posts WAV audio to the bundled Embeddable Lemonade (`lemond`) subprocess on
// its loopback port. Lemonade exposes an OpenAI-compatible
// `/api/v1/audio/transcriptions` endpoint, so this mirrors the cloud Groq
// provider almost exactly — only the base URL, model name, and per-launch
// bearer key differ. Runs on the NPU via recipe `flm`, model
// `whisper-v3-turbo-FLM`.

use reqwest::multipart;
use serde::Deserialize;
use std::sync::OnceLock;

use crate::services::lemond::LOCAL_MODEL;

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn get_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .pool_max_idle_per_host(4)
            .build()
            .expect("Failed to create local HTTP client")
    })
}

#[derive(Debug, Deserialize)]
struct WhisperResponse {
    text: String,
}

/// Transcribe an audio segment using the local lemond NPU endpoint.
pub async fn transcribe_verbose(
    audio_data: Vec<u8>,
    port: u16,
    key: String,
    language: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    if audio_data.len() < 100 {
        return Err("Audio data too small".into());
    }

    let client = get_http_client();

    let part = multipart::Part::bytes(audio_data)
        .file_name("output.wav")
        .mime_str("audio/wav")?;

    let mut form = multipart::Form::new()
        .part("file", part)
        .text("model", LOCAL_MODEL)
        .text("response_format", "verbose_json");

    if let Some(lang) = language {
        form = form.text("language", lang);
    }

    let url = format!("http://127.0.0.1:{port}/api/v1/audio/transcriptions");
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                "Request timeout".into()
            } else if e.is_connect() {
                "Local engine not reachable".into()
            } else {
                Box::new(e) as Box<dyn std::error::Error>
            }
        })?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Local engine error ({}): {}", status.as_u16(), error_text).into());
    }

    let result: WhisperResponse = response.json().await?;
    Ok(result.text)
}
