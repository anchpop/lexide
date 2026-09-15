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
        let mut body = vec![0; length.unwrap()];
        reader.read_exact(&mut body).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            expected_request
        );
        let body = response.to_string();
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
