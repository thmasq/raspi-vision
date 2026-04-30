#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]

mod apriltag;

use crate::apriltag::{
    decode::{AprilTagDetection, QuickDecode, extract_detection},
    quad::find_quad_corners,
};
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
use serde::{Deserialize, Serialize};
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

// --- CAMERA CONTROLS ---
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum DebugView {
    Raw,
    Threshold,
    Segments,
}

#[derive(Clone, Debug, Deserialize)]
struct CameraControls {
    ae_enable: bool,
    exposure_time: i32,
    analogue_gain: f32,
    debug_view: DebugView,
}

struct AppState {
    tx_video: broadcast::Sender<Vec<u8>>,
    tx_tags: broadcast::Sender<String>,
    controls_tx: watch::Sender<CameraControls>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let (tx_video, _) = broadcast::channel::<Vec<u8>>(16);
    let (tx_tags, _) = broadcast::channel::<String>(16);

    let (controls_tx, controls_rx) = watch::channel(CameraControls {
        ae_enable: false,
        exposure_time: 30000,
        analogue_gain: 7.0,
        debug_view: DebugView::Raw,
    });

    let app_state = Arc::new(AppState {
        tx_video: tx_video.clone(),
        tx_tags: tx_tags.clone(),
        controls_tx,
    });

    let capture_controls_rx = controls_rx.clone();
    tokio::task::spawn_blocking(move || {
        capture_loop(&tx_video, &tx_tags, &capture_controls_rx);
    });

    let app = Router::new()
        .route("/", get(index_html))
        .route("/ws", get(websocket_handler))
        .route("/controls", post(update_controls))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 9080));
    println!("Server running at http://{addr}");

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
    tx_video: &broadcast::Sender<Vec<u8>>,
    tx_tags: &broadcast::Sender<String>,
    controls_rx: &watch::Receiver<CameraControls>,
) {
    let mgr = CameraManager::new().expect("Failed to initialize libcamera");
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("No cameras found");
    let mut camera = cam.acquire().expect("Failed to acquire camera");

    let mut config = camera
        .generate_configuration(&[
            libcamera::stream::StreamRole::VideoRecording,
            libcamera::stream::StreamRole::ViewFinder,
        ])
        .unwrap();

    {
        let mut stream_cfg = config.get_mut(0).unwrap();
        stream_cfg.set_size(libcamera::geometry::Size {
            width: CAPTURE_WIDTH as u32,
            height: CAPTURE_HEIGHT as u32,
        });
        stream_cfg.set_pixel_format(formats::YUV420);
    }

    {
        let mut stream_cfg = config.get_mut(1).unwrap();
        stream_cfg.set_size(libcamera::geometry::Size {
            width: STREAM_WIDTH as u32,
            height: STREAM_HEIGHT as u32,
        });
        stream_cfg.set_pixel_format(formats::YUV420);
    }

    config.validate();

    let capture_stride = config.get(0).unwrap().get_stride() as usize;
    let vga_stride = config.get(1).unwrap().get_stride() as usize;

    camera
        .configure(&mut config)
        .expect("Failed to configure camera");

    let mut allocator = libcamera::framebuffer_allocator::FrameBufferAllocator::new(&camera);
    let stream_full = config.get(0).unwrap().stream().unwrap().clone();
    let stream_vga = config.get(1).unwrap().stream().unwrap().clone();

    let allocated_buffers_full = allocator.alloc(&stream_full).unwrap();
    let allocated_buffers_vga = allocator.alloc(&stream_vga).unwrap();

    let requests: Vec<_> = allocated_buffers_full
        .into_iter()
        .zip(allocated_buffers_vga.into_iter())
        .map(|(buf_full, buf_vga)| {
            let mut request = camera.create_request(None).unwrap();

            let mapped_full = MemoryMappedFrameBuffer::new(buf_full).unwrap();
            request.add_buffer(&stream_full, mapped_full).unwrap();

            let mapped_vga = MemoryMappedFrameBuffer::new(buf_vga).unwrap();
            request.add_buffer(&stream_vga, mapped_vga).unwrap();

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

    println!(
        "Starting dual-stream capture loop: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} and {STREAM_WIDTH}x{STREAM_HEIGHT}..."
    );

    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<(Vec<u8>, Vec<u8>)>(1);

    let tx_video_pipe = tx_video.clone();
    let tx_tags_pipe = tx_tags.clone();
    let controls_rx_pipe = controls_rx.clone();

    std::thread::spawn(move || {
        let mut global_uf =
            crate::apriltag::unionfind::UnionFind::new((CAPTURE_WIDTH * CAPTURE_HEIGHT) as u32);
        let mut mono_image =
            crate::apriltag::image::Image::new_simd_aligned(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        let mut threshold_img =
            crate::apriltag::image::Image::new_simd_aligned(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        let intrinsics = crate::apriltag::pose::CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            tag_size_mm: 165.0,
        };

        let quick_decode = QuickDecode::new();

        let mut vga_lut_dense: Vec<usize> = Vec::with_capacity(STREAM_WIDTH * STREAM_HEIGHT);
        for y in 0..STREAM_HEIGHT {
            let src_y = y * CAPTURE_HEIGHT / STREAM_HEIGHT;
            for x in 0..STREAM_WIDTH {
                let src_x = x * CAPTURE_WIDTH / STREAM_WIDTH;
                vga_lut_dense.push(src_y * CAPTURE_WIDTH + src_x);
            }
        }

        let mut frames_processed = 0;
        let mut pipe_fps_timer = Instant::now();

        for (raw_frame_full, raw_frame_vga) in frame_rx {
            frames_processed += 1;
            if pipe_fps_timer.elapsed() >= Duration::from_secs(1) {
                println!("Pipeline FPS: {frames_processed}");
                frames_processed = 0;
                pipe_fps_timer = Instant::now();
            }

            let pipe_start = Instant::now();
            let current_controls = controls_rx_pipe.borrow().clone();

            let t_start = Instant::now();
            let mono_slice = mono_image.as_mut_slice();

            if capture_stride == CAPTURE_WIDTH {
                mono_slice.copy_from_slice(&raw_frame_full[..CAPTURE_WIDTH * CAPTURE_HEIGHT]);
            } else {
                for y in 0..CAPTURE_HEIGHT {
                    let src_offset = y * capture_stride;
                    let dst_offset = y * CAPTURE_WIDTH;
                    mono_slice[dst_offset..dst_offset + CAPTURE_WIDTH]
                        .copy_from_slice(&raw_frame_full[src_offset..src_offset + CAPTURE_WIDTH]);
                }
            }

            let t_downsample = t_start.elapsed();

            let t_start = Instant::now();
            unsafe { crate::apriltag::threshold::process(&mono_image, &mut threshold_img) };
            let t_thresh = t_start.elapsed();

            let t_start = Instant::now();
            global_uf.clear();
            global_uf.connected_components(threshold_img.as_slice(), CAPTURE_WIDTH, CAPTURE_HEIGHT);
            global_uf.flatten();
            let t_uf = t_start.elapsed();

            let t_start = Instant::now();
            let clusters = global_uf.gradient_clusters(
                threshold_img.as_slice(),
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT,
            );
            let t_cluster = t_start.elapsed();

            let debug_frame = match current_controls.debug_view {
                DebugView::Raw => {
                    let mut buf = vec![0u8; STREAM_WIDTH * STREAM_HEIGHT];
                    if vga_stride == STREAM_WIDTH {
                        buf.copy_from_slice(&raw_frame_vga[..STREAM_WIDTH * STREAM_HEIGHT]);
                    } else {
                        for y in 0..STREAM_HEIGHT {
                            let src_offset = y * vga_stride;
                            let dst_offset = y * STREAM_WIDTH;
                            buf[dst_offset..dst_offset + STREAM_WIDTH].copy_from_slice(
                                &raw_frame_vga[src_offset..src_offset + STREAM_WIDTH],
                            );
                        }
                    }
                    buf
                }
                DebugView::Threshold => {
                    let mut buf = vec![0u8; STREAM_WIDTH * STREAM_HEIGHT];
                    let thresh_slice = threshold_img.as_slice();
                    for (dst, &src_idx) in buf.iter_mut().zip(&vga_lut_dense) {
                        *dst = thresh_slice[src_idx];
                    }
                    buf
                }
                DebugView::Segments => {
                    let mut buf = vec![0u8; STREAM_WIDTH * STREAM_HEIGHT];
                    let scale_x = CAPTURE_WIDTH as f32 / STREAM_WIDTH as f32;
                    let scale_y = CAPTURE_HEIGHT as f32 / STREAM_HEIGHT as f32;

                    for cluster in &clusters {
                        for pt in &cluster.points {
                            let px = ((f32::from(pt.x) / 2.0) / scale_x) as usize;
                            let py = ((f32::from(pt.y) / 2.0) / scale_y) as usize;
                            let idx = py * STREAM_WIDTH + px;
                            if idx < buf.len() {
                                buf[idx] = 255;
                            }
                        }
                    }
                    buf
                }
            };

            let _ = tx_video_pipe.send(debug_frame);

            let t_start = Instant::now();
            let mut valid_detections: Vec<_> = clusters
                .iter()
                .filter_map(|cluster| {
                    let corners = find_quad_corners(cluster)?;
                    extract_detection(&mono_image, &corners, &intrinsics, &quick_decode)
                })
                .collect();
            let t_decode = t_start.elapsed();

            valid_detections.sort_unstable_by(|a, b| {
                let conf_a = a.confidence;
                let conf_b = b.confidence;
                conf_b.partial_cmp(&conf_a).unwrap()
            });

            let mut filtered_detections = Vec::new();

            for det in valid_detections {
                let is_duplicate = filtered_detections.iter().any(|kept: &AprilTagDetection| {
                    let dx = kept.center_x - det.center_x;
                    let dy = kept.center_y - det.center_y;
                    let dist_sq = dy.mul_add(dy, dx * dx);

                    dist_sq < 100.0
                });

                if !is_duplicate {
                    filtered_detections.push(det);
                }
            }

            let total_pipe = pipe_start.elapsed();

            println!(
                "Pipe: {total_pipe:?} | HW Downsample: {t_downsample:?}, Thresh: {t_thresh:?}, UF: {t_uf:?}, Cluster: {t_cluster:?}, Decode: {t_decode:?}"
            );

            let top_detections: Vec<_> = filtered_detections.into_iter().take(10).collect();
            if !top_detections.is_empty() {
                println!(
                    "Found {} tag(s): {:?}",
                    top_detections.len(),
                    top_detections
                );
                if let Ok(json) = serde_json::to_string(&top_detections) {
                    let _ = tx_tags_pipe.send(json);
                }
            }
        }
    });

    let mut fps_timer = Instant::now();
    let mut frames_captured = 0;

    loop {
        let mut req = cam_rx.recv().unwrap();

        let buffer_full: &MemoryMappedFrameBuffer<FrameBuffer> =
            req.buffer(&stream_full).expect("Failed to get full buffer");
        let planes_data_full = buffer_full.data();
        let raw_data_full = planes_data_full
            .first()
            .expect("Full buffer has no plane data");

        let buffer_vga: &MemoryMappedFrameBuffer<FrameBuffer> =
            req.buffer(&stream_vga).expect("Failed to get VGA buffer");
        let planes_data_vga = buffer_vga.data();
        let raw_data_vga = planes_data_vga
            .first()
            .expect("VGA buffer has no plane data");

        frames_captured += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            println!("Capture FPS: {frames_captured}");
            frames_captured = 0;
            fps_timer = Instant::now();
        }

        let frame_copy_full = raw_data_full.to_vec();
        let frame_copy_vga = raw_data_vga.to_vec();

        match frame_tx.try_send((frame_copy_full, frame_copy_vga)) {
            Ok(_) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {}
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                eprintln!("Pipeline thread panicked or disconnected!");
                break;
            }
        }

        req.reuse(libcamera::request::ReuseFlag::REUSE_BUFFERS);

        let current_controls = controls_rx.borrow().clone();
        let controls = req.controls_mut();

        controls
            .set(controls::AeEnable(current_controls.ae_enable))
            .unwrap();

        controls
            .set(controls::ExposureTime(current_controls.exposure_time))
            .unwrap();
        controls
            .set(controls::AnalogueGain(current_controls.analogue_gain))
            .unwrap();

        controls.set(controls::AwbEnable(false)).unwrap();
        controls.set(controls::Saturation(0.0)).unwrap();
        controls.set(controls::Sharpness(0.0)).unwrap();
        controls.set(controls::NoiseReductionMode::Off).unwrap();

        camera.queue_request(req).unwrap();
    }
}
