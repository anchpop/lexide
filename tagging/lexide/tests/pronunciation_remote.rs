#![cfg(feature = "pronunciation-remote")]

use std::future::Future;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use lexide::pronunciation::{remote::PhonemizerClient, ModelIdentity, PredictRequest};
use serde_json::{json, Value};

// One loopback request per test case. Check the real method, path, body, and
// custom client header without contacting Modal or performing inference.
fn with_response<F: Future<Output = ()>>(
    path: &'static str,
    expected_request: Value,
    response: Value,
    test: impl FnOnce(PhonemizerClient) -> F,
) {
    with_status_response(path, expected_request, 200, response, test)
}

fn with_status_response<F: Future<Output = ()>>(
    path: &'static str,
    expected_request: Value,
    status: u16,
    response: Value,
    test: impl FnOnce(PhonemizerClient) -> F,
) {
    with_literal_response(path, expected_request, status, response.to_string(), test)
}

fn with_literal_response<F: Future<Output = ()>>(
    path: &'static str,
    expected_request: Value,
    status: u16,
    body: String,
    test: impl FnOnce(PhonemizerClient) -> F,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, format!("POST {path} HTTP/1.1\r\n"));
        let mut length = None;
        let mut custom_client = false;
        loop {
            line.clear();
            assert_ne!(reader.read_line(&mut line).unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            let (name, value) = line.trim().split_once(':').unwrap();
            if name.eq_ignore_ascii_case("content-length") {
                length = Some(value.trim().parse::<usize>().unwrap());
            }
            if name.eq_ignore_ascii_case("x-test-client") {
                custom_client = value.trim() == "replacement";
            }
        }
        assert!(
            custom_client,
            "with_http_client must use the supplied client"
        );
        let mut request_body = vec![0; length.unwrap()];
        reader.read_exact(&mut request_body).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&request_body).unwrap(),
            expected_request
        );
        write!(stream, "HTTP/1.1 {status} Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    });
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("x-test-client", "replacement".parse().unwrap());
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .default_headers(headers)
        .build()
        .unwrap();
    let client = PhonemizerClient::with_endpoints(
        reqwest::Client::new(),
        format!("{url}/predict"),
        format!("{url}/batch"),
    )
    .unwrap()
    .with_http_client(http);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(test(client));
    server.join().unwrap();
}

#[test]
fn identity_accepts_valid_fields_and_unknown_extensions() {
    let response = json!({"model_id": "m", "model_revision": "r",
        "deploy_marker": "test", "decoder_version": "nonblank_v1",
        "language_head_specs": {}});
    let expected: ModelIdentity = serde_json::from_value(response.clone()).unwrap();
    with_response(
        "/predict",
        json!({"marker_only": true}),
        response,
        |client| async move {
            assert_eq!(client.identity().await.unwrap(), expected);
        },
    );
}

#[test]
fn identity_rejects_load_error_even_with_valid_model_fields() {
    // Presence is an error, including null or an unexpected structured value.
    for load_error in [
        json!("weights failed to load"),
        json!(null),
        json!({"reason": "missing weights"}),
    ] {
        for with_model_fields in [false, true] {
            let expected_error = load_error.to_string();
            let mut response = json!({"load_error": load_error});
            if with_model_fields {
                response["model_id"] = json!("m");
                response["model_revision"] = json!("r");
            }
            with_response(
                "/predict",
                json!({"marker_only": true}),
                response,
                |client| async move {
                    let error = client.identity().await.unwrap_err().to_string();
                    assert!(error.contains("load_error"), "{error}");
                    assert!(error.contains(&expected_error), "{error}");
                },
            );
        }
    }
}

#[test]
fn identity_requires_model_fields() {
    with_response(
        "/predict",
        json!({"marker_only": true}),
        json!({"deploy_marker": "test"}),
        |client| async move {
            let error = client.identity().await.unwrap_err();
            assert!(format!("{error:#}").contains("missing field `model_id`"));
        },
    );
}

#[test]
fn predict_batch_rejects_result_count_mismatch() {
    let request = PredictRequest::from_samples(&[0.5]);
    with_response(
        "/batch",
        json!({"requests": [request]}),
        json!({"results": []}),
        |client| async move {
            let error = client.predict_batch(&[request]).await.unwrap_err();
            assert_eq!(error.to_string(), "batch returned 0 results for 1 clips");
        },
    );
}

#[test]
fn error_status_reports_the_body_and_preserves_the_status() {
    let request = PredictRequest::from_samples(&[0.5]);
    let expected = serde_json::to_value(&request).unwrap();
    with_status_response(
        "/predict",
        expected,
        422,
        json!({"detail": "clip exceeds the maximum length"}),
        |client| async move {
            let error = client.predict(&request).await.unwrap_err();
            let rendered = format!("{error:#}");
            // The endpoint's reason is what makes a rejection diagnosable.
            assert!(
                rendered.contains("clip exceeds the maximum length"),
                "{rendered}"
            );
            assert!(rendered.contains("422"), "{rendered}");
            // Callers classify retries by recovering the status from the chain.
            let status = error
                .chain()
                .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
                .find_map(reqwest::Error::status);
            assert_eq!(status, Some(reqwest::StatusCode::UNPROCESSABLE_ENTITY));
        },
    );
}

#[test]
fn prediction_preserves_raw_bytes_and_parses_extensions() {
    let body = " \n{\"phonemes\":[{\"phoneme\":\"\\u0061\",\"confidence\":1.250e-2}],\"future\": { \"value\": 3.00 }}\t\n";
    for raw in [false, true] {
        let request = PredictRequest::default();
        with_literal_response(
            "/predict",
            json!(request),
            200,
            body.to_owned(),
            |client| async move {
                let typed = if raw {
                    let bytes = client.predict_raw(&request).await.unwrap();
                    assert_eq!(bytes, body.as_bytes());
                    serde_json::from_slice::<lexide::pronunciation::PredictResponse>(&bytes)
                        .unwrap()
                } else {
                    client.predict(&request).await.unwrap()
                };
                assert_eq!(typed.phonemes[0].phoneme, "a");
                assert_eq!(typed.phonemes[0].confidence, 0.0125);
                assert!(typed.model_id.is_none());
            },
        );
    }
}

#[test]
fn batch_preserves_extensions_and_per_clip_envelope_without_siblings() {
    let first = r#"{ "phonemes":[], "future_matrix":"AAA\u0041", "score":1.250e-2 }"#;
    let second = r#"{"error":{"type":"ValueError","message":"bad\nclip"},"future":true}"#;
    let body = format!(
        " \n{{\"model_\\u0069d\":\"m\\u006fdel\",\"model_revision\":\"r\",\"results\" : [ \n {first} ,\t {second}\n ], \"future_envelope\":{{\"n\":2.00}},\"deploy_marker\":\"d\\u0065ploy\"}}\t\n"
    );
    for raw in [false, true] {
        let body = body.clone();
        let requests = vec![PredictRequest::default(); 2];
        with_literal_response(
            "/batch",
            json!({"requests": requests}),
            200,
            body.clone(),
            |client| async move {
                if !raw {
                    use lexide::pronunciation::BatchResult;
                    let typed = client.predict_batch(&requests).await.unwrap();
                    assert_eq!(typed.model_id.as_deref(), Some("model"));
                    assert_eq!(typed.model_revision.as_deref(), Some("r"));
                    assert_eq!(typed.deploy_marker.as_deref(), Some("deploy"));
                    assert!(typed.decoder_version.is_none());
                    assert_eq!(typed.results.len(), 2);
                    assert!(
                        matches!(&typed.results[0], BatchResult::Prediction(prediction) if prediction.phonemes.is_empty())
                    );
                    assert!(
                        matches!(&typed.results[1], BatchResult::Error { error } if error.message == "bad\nclip")
                    );
                    return;
                }
                let raw = client.predict_batch_raw(&requests).await.unwrap();
                assert_eq!(raw.batch.results[0].get(), first);
                assert_eq!(raw.batch.results[1].get(), second);
                assert_eq!(raw.batch.model_id.as_deref(), Some("model"));
                assert_eq!(raw.batch.model_revision.as_deref(), Some("r"));
                assert_eq!(raw.batch.deploy_marker.as_deref(), Some("deploy"));
                assert!(raw.batch.decoder_version.is_none());
                assert!(!raw.envelope.contains_key("results"));
                assert!(!raw.envelope.contains_key("decoder_version"));
                assert_eq!(raw.envelope["future_envelope"].get(), r#"{"n":2.00}"#);
                let identity: ModelIdentity = serde_json::from_slice(body.as_bytes()).unwrap();
                for item in &raw.batch.results {
                    let cached = serde_json::to_vec(&(&raw.envelope, item)).unwrap();
                    let (envelope, result): (
                        std::collections::BTreeMap<String, Box<serde_json::value::RawValue>>,
                        Box<serde_json::value::RawValue>,
                    ) = serde_json::from_slice(&cached).unwrap();
                    assert_eq!(result.get(), item.get());
                    assert_eq!(envelope.len(), raw.envelope.len());
                    for (key, value) in &raw.envelope {
                        assert_eq!(envelope[key].get(), value.get());
                    }
                    assert_eq!(
                        serde_json::from_slice::<ModelIdentity>(
                            &serde_json::to_vec(&envelope).unwrap()
                        )
                        .unwrap(),
                        identity
                    );
                    let cached_text = std::str::from_utf8(&cached).unwrap();
                    let sibling = if item.get() == first { second } else { first };
                    assert!(!cached_text.contains(sibling));
                }
            },
        );
    }
}

#[test]
fn batch_bounds_are_checked_before_sending() {
    let client = PhonemizerClient::with_endpoints(
        reqwest::Client::new(),
        "http://127.0.0.1:1/predict",
        "http://127.0.0.1:1/batch",
    )
    .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            for count in [0, 65] {
                let requests = vec![PredictRequest::default(); count];
                for error in [
                    client.predict_batch_raw(&requests).await.unwrap_err(),
                    client.predict_batch(&requests).await.unwrap_err(),
                ] {
                    assert_eq!(
                        error.to_string(),
                        "requests must contain between 1 and 64 items"
                    );
                }
            }
        });
}

#[test]
fn raw_and_typed_batch_failures_preserve_decoding_before_cardinality() {
    for raw in [false, true] {
        for body in [
            r#"{"results":[]}"#,
            r#"{"results":[{},{}]}"#,
            r#"{"results":[{},{}],"model_id":42}"#,
            r#"{"model_id":42,"results":[]}"#,
            r#"{"results":null}"#,
            r#"{"results":[],"results":[]}"#,
            r#"{"results": ["#,
        ] {
            let request = PredictRequest::default();
            with_literal_response(
                "/batch",
                json!({"requests":[request]}),
                200,
                body.to_owned(),
                |client| async move {
                    let error = if raw {
                        client.predict_batch_raw(&[request]).await.unwrap_err()
                    } else {
                        client.predict_batch(&[request]).await.unwrap_err()
                    };
                    if body == r#"{"results":[]}"# {
                        assert_eq!(error.to_string(), "batch returned 0 results for 1 clips");
                    } else {
                        assert_eq!(error.to_string(), "invalid batch response");
                        let expected =
                            serde_json::from_str::<lexide::pronunciation::BatchResponse>(body)
                                .unwrap_err();
                        let actual = error
                            .chain()
                            .find_map(|cause| cause.downcast_ref::<serde_json::Error>())
                            .unwrap();
                        assert_eq!(actual.to_string(), expected.to_string());
                        assert!(error.downcast_ref::<reqwest::Error>().unwrap().is_decode());
                        assert!(error.chain().any(|cause| cause.is::<serde_json::Error>()));
                    }
                },
            );
        }
    }
}

#[test]
fn raw_keeps_future_items_while_typed_predictors_reject_malformed_fields() {
    let request = PredictRequest::default();
    with_literal_response(
        "/batch",
        json!({"requests":[request]}),
        200,
        r#"{"results":[{"future_prediction":true}]}"#.to_owned(),
        |client| async move {
            assert!(client.predict_batch_raw(&[request]).await.is_ok());
        },
    );
    for batch in [false, true] {
        let request = PredictRequest::default();
        let (path, expected, body, context) = if batch {
            (
                "/batch",
                json!({"requests":[request]}),
                r#"{"results":[{}]}"#,
                "invalid batch response",
            )
        } else {
            (
                "/predict",
                json!(request),
                "{}",
                "invalid prediction response",
            )
        };
        with_literal_response(path, expected, 200, body.to_owned(), |client| async move {
            let error = if batch {
                client.predict_batch(&[request]).await.unwrap_err()
            } else {
                client.predict(&request).await.unwrap_err()
            };
            assert_eq!(error.to_string(), context);
            assert!(error.downcast_ref::<reqwest::Error>().unwrap().is_decode());
            assert!(error.chain().any(|cause| cause.is::<serde_json::Error>()));
        });
    }
}

#[test]
fn raw_predictors_preserve_http_errors() {
    for batch in [false, true] {
        let request = PredictRequest::default();
        let (path, expected) = if batch {
            ("/batch", json!({"requests": [request]}))
        } else {
            ("/predict", json!(request))
        };
        with_status_response(
            path,
            expected,
            422,
            json!({"detail":"rejected clip"}),
            |client| async move {
                let error = if batch {
                    client.predict_batch_raw(&[request]).await.unwrap_err()
                } else {
                    client.predict_raw(&request).await.unwrap_err()
                };
                assert!(error.to_string().contains("rejected clip"));
                assert_eq!(
                    error.downcast_ref::<reqwest::Error>().unwrap().status(),
                    Some(reqwest::StatusCode::UNPROCESSABLE_ENTITY)
                );
            },
        );
    }
}

#[test]
fn batch_accepts_the_upper_bound_and_optional_identity() {
    use lexide::pronunciation::BatchResult;
    for raw in [false, true] {
        let requests = vec![PredictRequest::default(); 64];
        let mut results = vec![json!({"phonemes":[]}); 64];
        results[63] = json!({"error":{"type":"ValueError","message":"bad clip"}});
        with_response(
            "/batch",
            json!({"requests":requests}),
            json!({"results":results}),
            |client| async move {
                if raw {
                    let response = client.predict_batch_raw(&requests).await.unwrap();
                    assert_eq!(response.batch.results.len(), 64);
                    assert!(response.batch.model_id.is_none());
                    assert!(response.envelope.is_empty());
                } else {
                    let response = client.predict_batch(&requests).await.unwrap();
                    assert_eq!(response.results.len(), 64);
                    assert!(response.model_id.is_none());
                    assert!(matches!(response.results[63], BatchResult::Error { .. }));
                }
            },
        );
    }
}

#[tokio::test]
async fn arbitrary_batches_compact_failures_retry_and_keep_ids_and_metadata() {
    use futures::StreamExt;
    use lexide::pronunciation::remote::{AudioClip, AudioInput, RequestActivity};
    use std::sync::Arc;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/batch", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let mut sizes = Vec::new();
        while sizes.len() < 4 {
            let (mut socket, _) = match listener.accept() {
                Ok(socket) => socket,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "missing batch request"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
                Err(e) => panic!("{e}"),
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&socket);
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((key, value)) = line.split_once(':') {
                    if key.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                }
            }
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            let requests = body["requests"].as_array().unwrap();
            sizes.push(requests.len());
            let (status, response) = if sizes.len() == 1 {
                (503, json!({"detail": "cold"}))
            } else {
                (
                    200,
                    json!({"future_field": {"kept": 42}, "results": requests.iter().map(|request| {
                    if request["top_k"] == 7 { json!({"error": {"type": "ValueError", "message": "one bad clip"}}) }
                    else { json!({"phonemes": [], "clip_id": request["top_k"]}) }
                }).collect::<Vec<_>>()}),
                )
            };
            let body = response.to_string();
            write!(socket, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
        sizes
    });
    let activity = Arc::new(RequestActivity::default());
    let client = PhonemizerClient::with_endpoints(
        reqwest::Client::builder().no_proxy().build().unwrap(),
        &url,
        &url,
    )
    .unwrap()
    .with_activity(activity.clone());
    let mut clips: Vec<_> = (0..130)
        .rev()
        .map(|id| AudioClip {
            id,
            duration: Duration::from_millis(id),
            audio: AudioInput::Request(PredictRequest {
                top_k: id as usize,
                ..PredictRequest::from_samples(&[0.0])
            }),
        })
        .collect();
    clips.push(AudioClip {
        id: 130,
        duration: Duration::ZERO,
        audio: AudioInput::File("/missing/lexide-batch-fixture.wav".into()),
    });
    let mut results = tokio::time::timeout(
        Duration::from_secs(20),
        client.predict_many(clips).collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    results.sort_by_key(|(id, _)| *id);
    assert_eq!(results.len(), 131);
    for (id, result) in results {
        if id == 7 || id == 130 {
            assert!(result.is_err());
            continue;
        }
        let result = result.unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(result.item.get()).unwrap()["clip_id"],
            id
        );
        assert_eq!(
            serde_json::from_str::<Value>(result.envelope["future_field"].get()).unwrap(),
            json!({"kept": 42})
        );
    }
    let mut sizes = server.join().unwrap();
    sizes.sort();
    assert_eq!(sizes, [2, 64, 64, 64]);
    assert_eq!(activity.snapshot().retries, 1);
    assert!(activity.snapshot().peak_requests <= 2);
}

#[test]
fn file_and_byte_callers_share_the_individual_queue() {
    use futures::future::join;
    use lexide::pronunciation::remote::{request_from_samples, AudioInput};
    let wav = b"RIFF\x26\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\x02\0\0\0\0\0";
    let path = std::env::temp_dir().join(format!("lexide-batch-{}.wav", std::process::id()));
    std::fs::write(&path, wav).unwrap();
    let request = request_from_samples(&[0.0], 16_000, 10);
    with_response(
        "/batch",
        json!({"requests": [request, request]}),
        json!({"results": [{"phonemes": []}, {"phonemes": []}]}),
        |client| {
            let path = path.clone();
            async move {
                let (a, b) = join(
                    client.predict_audio(AudioInput::File(path)),
                    client.predict_audio(AudioInput::Bytes(wav.to_vec())),
                )
                .await;
                assert!(a.is_ok(), "{a:?}");
                assert!(b.is_ok(), "{b:?}");
            }
        },
    );
    std::fs::remove_file(path).unwrap();
}
