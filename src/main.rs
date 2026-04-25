mod apriltag;

use axum::{
    Json, Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::{Html, IntoResponse},
    routing::{get, post},
};
use libcamera::framebuffer_allocator::FrameBuffer;
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::{camera_manager::CameraManager, controls, formats};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, watch};

// --- CAPTURE SETTINGS ---
const CAPTURE_WIDTH: usize = 1296;
const CAPTURE_HEIGHT: usize = 972;

// --- STREAM SETTINGS ---
const STREAM_WIDTH: usize = 640;
const STREAM_HEIGHT: usize = 480;
const STREAM_EVERY_N_FRAMES: usize = 3;

// --- CAMERA CONTROLS ---
#[derive(Clone, Debug, Deserialize)]
struct CameraControls {
    ae_enable: bool,
    exposure_time: i32,
    analogue_gain: f32,
}

struct AppState {
    tx_video: broadcast::Sender<Vec<u8>>,
    tx_tags: broadcast::Sender<String>,
    controls_tx: watch::Sender<CameraControls>,
}

#[tokio::main]
async fn main() {
    let (tx_video, _) = broadcast::channel::<Vec<u8>>(16);
    let (tx_tags, _) = broadcast::channel::<String>(16);

    let (controls_tx, controls_rx) = watch::channel(CameraControls {
        ae_enable: true,
        exposure_time: 20000,
        analogue_gain: 1.0,
    });

    let app_state = Arc::new(AppState {
        tx_video: tx_video.clone(),
        tx_tags: tx_tags.clone(),
        controls_tx,
    });

    let capture_controls_rx = controls_rx.clone();
    tokio::task::spawn_blocking(move || {
        capture_loop(tx_video, tx_tags, capture_controls_rx);
    });

    let app = Router::new()
        .route("/", get(index_html))
        .route("/ws", get(websocket_handler))
        .route("/controls", post(update_controls))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 9080));
    println!("Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn index_html() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx_video = state.tx_video.subscribe();
    let mut rx_tags = state.tx_tags.subscribe();

    loop {
        tokio::select! {
            Ok(frame_data) = rx_video.recv() => {
                if socket.send(Message::Binary(frame_data.into())).await.is_err() {
                    break;
                }
            }
            Ok(tags_json) = rx_tags.recv() => {
                if socket.send(Message::Text(tags_json.into())).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn update_controls(
    State(state): State<Arc<AppState>>,
    Json(controls): Json<CameraControls>,
) -> impl IntoResponse {
    let _ = state.controls_tx.send(controls);
    axum::http::StatusCode::OK
}

fn capture_loop(
    tx_video: broadcast::Sender<Vec<u8>>,
    tx_tags: broadcast::Sender<String>,
    controls_rx: watch::Receiver<CameraControls>,
) {
    let mgr = CameraManager::new().expect("Failed to initialize libcamera");
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("No cameras found");
    let mut camera = cam.acquire().expect("Failed to acquire camera");

    let mut config = camera
        .generate_configuration(&[libcamera::stream::StreamRole::VideoRecording])
        .unwrap();
    {
        let mut stream_cfg = config.get_mut(0).unwrap();
        stream_cfg.set_size(libcamera::geometry::Size {
            width: CAPTURE_WIDTH as u32,
            height: CAPTURE_HEIGHT as u32,
        });
        stream_cfg.set_pixel_format(formats::YUV420);
    }

    config.validate();
    let capture_stride = config.get(0).unwrap().get_stride() as usize;

    camera
        .configure(&mut config)
        .expect("Failed to configure camera");

    let mut allocator = libcamera::framebuffer_allocator::FrameBufferAllocator::new(&camera);
    let stream = config.get(0).unwrap().stream().unwrap();
    let allocated_buffers = allocator.alloc(&stream).unwrap();

    let requests: Vec<_> = allocated_buffers
        .into_iter()
        .map(|buffer| {
            let mut request = camera.create_request(None).unwrap();
            let mapped_buffer = MemoryMappedFrameBuffer::new(buffer).unwrap();
            request.add_buffer(&stream, mapped_buffer).unwrap();
            request
        })
        .collect();

    camera.start(None).unwrap();

    for request in requests {
        camera.queue_request(request).unwrap();
    }

    let (cam_tx, cam_rx) = std::sync::mpsc::channel();
    camera.on_request_completed(move |req| {
        cam_tx.send(req).unwrap();
    });

    let mut global_uf = crate::apriltag::unionfind::RleUnionFind::new(8000);
    let mut mono_image =
        crate::apriltag::image::Image::new_simd_aligned(STREAM_WIDTH, STREAM_HEIGHT);
    let mut threshold_img =
        crate::apriltag::image::Image::new_simd_aligned(STREAM_WIDTH, STREAM_HEIGHT);
    let intrinsics = crate::apriltag::pose::CameraIntrinsics {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        tag_size_mm: 165.0,
    };

    println!(
        "Starting capture loop at {}x{}...",
        CAPTURE_WIDTH, CAPTURE_HEIGHT
    );

    let mut fps_timer = Instant::now();
    let mut frames_captured = 0;
    let mut total_frames = 0;

    loop {
        let mut req = cam_rx.recv().unwrap();

        let buffer: &MemoryMappedFrameBuffer<FrameBuffer> =
            req.buffer(&stream).expect("Failed to get buffer");
        let planes_data = buffer.data();
        let raw_data = planes_data.get(0).expect("Buffer has no plane data");

        frames_captured += 1;
        total_frames += 1;

        if fps_timer.elapsed() >= Duration::from_secs(1) {
            println!("Capture FPS: {}", frames_captured);
            frames_captured = 0;
            fps_timer = Instant::now();
        }

        if total_frames % STREAM_EVERY_N_FRAMES == 0 {
            let mut stream_bytes = vec![0u8; STREAM_WIDTH * STREAM_HEIGHT];
            for y in 0..STREAM_HEIGHT {
                let src_y = y * CAPTURE_HEIGHT / STREAM_HEIGHT;
                for x in 0..STREAM_WIDTH {
                    let src_x = x * CAPTURE_WIDTH / STREAM_WIDTH;
                    let src_idx = src_y * capture_stride + src_x;
                    if src_idx < raw_data.len() {
                        stream_bytes[y * STREAM_WIDTH + x] = raw_data[src_idx];
                    }
                }
            }

            let _ = tx_video.send(stream_bytes.clone());

            mono_image.as_mut_slice().copy_from_slice(&stream_bytes);
            crate::apriltag::threshold::process(&mono_image, &mut threshold_img);

            global_uf.clear();
            global_uf.process_chunk(threshold_img.as_mut_slice(), STREAM_WIDTH, STREAM_HEIGHT, 0);
            global_uf.flatten();
            let blobs = global_uf.extract_valid_blobs(50, 10_000);

            let mut valid_detections = Vec::new();

            for blob in &blobs {
                let boundary =
                    crate::apriltag::quad::extract_ordered_boundary(blob, &global_uf.segments);

                if let Some(corners) = crate::apriltag::quad::find_quad_corners(&boundary) {
                    if let Some(detection) = crate::apriltag::decode::extract_detection(
                        &mono_image,
                        &corners,
                        &intrinsics,
                    ) {
                        valid_detections.push(detection);
                        if valid_detections.len() >= 10 {
                            break;
                        }
                    }
                }
            }

            if !valid_detections.is_empty() {
                if let Ok(json) = serde_json::to_string(&valid_detections) {
                    let _ = tx_tags.send(json);
                }
            }
        }

        req.reuse(libcamera::request::ReuseFlag::REUSE_BUFFERS);

        let current_controls = controls_rx.borrow().clone();

        let controls = req.controls_mut();

        controls
            .set(controls::AeEnable(current_controls.ae_enable))
            .unwrap();

        if !current_controls.ae_enable {
            controls
                .set(controls::ExposureTime(current_controls.exposure_time))
                .unwrap();
            controls
                .set(controls::AnalogueGain(current_controls.analogue_gain))
                .unwrap();
        }

        camera.queue_request(req).unwrap();
    }
}
