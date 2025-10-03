use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ndarray::{Array4, ArrayViewD, CowArray, IxDyn};
use once_cell::sync::Lazy;
use opencv::{
    core::{copy_make_border, Mat, Rect, Scalar, Size, CV_32F, BORDER_CONSTANT},
    highgui, imgproc, prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use ort::{environment::Environment, session::Session, session::SessionBuilder, value::Value};
use serde::Serialize;
use sqlx::{mysql::MySqlPoolOptions, MySql, Pool};
use std::{
    collections::VecDeque, fs, sync::Mutex,
    time::{Duration, Instant},
};
use time::OffsetDateTime;

/* ===================== LOGGER ===================== */
const C_RESET: &str = "\x1b[0m";
const C_OK: &str = "\x1b[32m";
const C_WARN: &str = "\x1b[33m";
const C_ERR: &str = "\x1b[31m";
fn ts() -> String {
    let now = OffsetDateTime::now_utc();
    format!("{:02}:{:02}:{:02}", now.hour(), now.minute(), now.second())
}
fn log_ok(tag: &str, msg: &str) { println!("[{}] [{}] {}✓{} {}", ts(), tag, C_OK, C_RESET, msg); }
fn log_warn(tag: &str, msg: &str) { eprintln!("[{}] [{}] {}WARN{}  {}", ts(), tag, C_WARN, C_RESET, msg); }
fn log_err(tag: &str, msg: &str) { eprintln!("[{}] [{}] {}ERR{}   {}", ts(), tag, C_ERR, C_RESET, msg); }

/* ===================== PREVIEW (1 window) ===================== */
const WIN_PREVIEW: &str = "Preview";
static PREVIEW_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
fn show_preview(img: &Mat) {
    let _g = PREVIEW_LOCK.lock().unwrap();
    let _ = highgui::named_window(WIN_PREVIEW, highgui::WINDOW_NORMAL);
    let _ = highgui::imshow(WIN_PREVIEW, img);
    let _ = highgui::wait_key(1);
}

/* ===================== CLI ===================== */
#[derive(Parser, Debug, Clone)]
#[command(about = "Realtime vehicle counter (OpenCV/ONNX). EV detection OFF. MySQL optional. MQTT args kept but disabled.")]
struct Args {
    // Model
    #[arg(long, default_value = "models/best.onnx")]
    model: String,
    #[arg(long, default_value = "models/classes.json")]
    classes: String,

    // Kamera (V4L2)
    #[arg(long, default_value_t = 0)]
    cam: i32,
    #[arg(long, default_value_t = 1280)]
    width: i32,
    #[arg(long, default_value_t = 720)]
    height: i32,
    #[arg(long, default_value_t = 30)]
    fps: i32,

    // Folder (kompatibel, tidak dipakai)
    #[arg(long, default_value = "videos")]
    video_dir: String,
    #[arg(long, default_value_t = 300)]
    segment_secs: u64,

    // Preview
    #[arg(long, default_value_t = false)]
    preview: bool,

    // Pemrosesan realtime
    #[arg(long, default_value_t = 10)]
    sample_fps: i32,
    #[arg(long, default_value_t = 640)]
    imgsz: i32,
    #[arg(long, default_value_t = 0.22)]
    conf_vehicle: f32,
    #[arg(long, default_value_t = 0.50)]
    conf_plate: f32,
    #[arg(long, default_value_t = 0.65)]
    nms_iou: f32,
    #[arg(long, default_value = "100,500,1180,500")]
    line: String, // referensi visual saja

    // Arsip (kompatibel)
    #[arg(long, default_value = "videos_done")]
    done_dir: String,
    #[arg(long, default_value = "research")]
    research_dir: String,

    // MQTT (kompatibel; tidak dipakai)
    #[arg(long, default_value = "your-broker.hivemq.cloud")]
    mqtt_host: String,
    #[arg(long, default_value = "user")]
    mqtt_user: String,
    #[arg(long, default_value = "pass")]
    mqtt_pass: String,
    #[arg(long, default_value = "traffic/siteA/cam01/clip_count")]
    mqtt_topic: String,
    #[arg(long, default_value = "veh-batch-01")]
    mqtt_client_id: String,

    // MySQL (opsional)
    #[arg(long, default_value = "")]
    db_url: String,
    #[arg(long, default_value_t = true)]
    db_enable: bool,

    #[arg(long, default_value_t = 5)]
    scan_interval: u64, // dummy

    /* == GPU options == */
    /// Prefer TensorRT EP (Jetson). Butuh ORT build dgn TensorRT.
    #[arg(long, default_value_t = false)]
    use_tensorrt: bool,
    /// Prefer CUDA EP. Butuh ORT build dgn CUDA.
    #[arg(long, default_value_t = false)]
    use_cuda: bool,
    /// Aktifkan FP16 TensorRT (kalau model dan device support)
    #[arg(long, default_value_t = false)]
    trt_fp16: bool,
    /// Folder cache engine TensorRT
    #[arg(long, default_value = "trt_cache")]
    trt_cache: String,
}

#[derive(Serialize)]
struct ClipCountMsg {
    ts: i64,
    cam_id: String,
    clip_path: String, // realtime: "rt-<ts>"
    count_non_ev: u32, // total kendaraan
    count_ev: u32,     // 0 (EV OFF)
    frames_processed: u32,
    sample_fps: i32,
}

/* ===================== MAIN ===================== */
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Siapkan folder (kompatibel)
    for d in [&args.video_dir, &args.done_dir, &args.research_dir, &args.trt_cache] {
        fs::create_dir_all(d).with_context(|| format!("make dir {}", d))?;
    }
    log_ok("BOOT", "dirs ready");

    /* -------- ONNX Runtime + GPU EP -------- */
    let env = Environment::builder().with_name("veh-rt").build()?.into_arc();

    // Siapkan SessionBuilder dengan optimasi maksimum
    let mut sb = SessionBuilder::new(&env)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?;

    // Kumpulkan providers sesuai flag, urutan = prioritas
    // Di Jetson: TensorRT biasanya paling kenceng; lalu CUDA; terakhir CPU
    #[allow(unused_mut)]
    let mut providers: Vec<ort::execution_providers::ExecutionProvider> = Vec::new();

    // TensorRT
    if args.use_tensorrt {
        #[cfg(feature = "tensorrt")]
        {
            use ort::execution_providers::{TensorRTExecutionProviderOptions, ExecutionProvider};
            let mut trt = TensorRTExecutionProviderOptions::default();
            trt.trt_fp16_enable = args.trt_fp16;
            trt.trt_engine_cache_enable = true;
            trt.trt_engine_cache_path = Some(args.trt_cache.clone());
            providers.push(ExecutionProvider::TensorRT(trt));
            log_info("ONNX", "Try TensorRT EP…");
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            log_warn("ONNX", "Binary ORT tidak di-compile dengan fitur 'tensorrt'; skip TensorRT EP");
        }
    }

    // CUDA
    if args.use_cuda {
        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::{CUDAExecutionProviderOptions, ExecutionProvider};
            // default: pakai device 0
            let cuda = CUDAExecutionProviderOptions {
                device_id: 0,
                cudnn_conv_algo_search: None,
                gpu_mem_limit: None, // pakai default ORT
                arena_extend_strategy: None,
                do_copy_in_default_stream: None,
                has_user_compute_stream: None,
                default_memory_arena_cfg: None,
            };
            providers.push(ExecutionProvider::CUDA(cuda));
            log_info("ONNX", "Try CUDA EP…");
        }
        #[cfg(not(feature = "cuda"))]
        {
            log_warn("ONNX", "Binary ORT tidak di-compile dengan fitur 'cuda'; skip CUDA EP");
        }
    }

    // Daftarkan providers kalau ada; kalau tidak ada, CPU saja.
    if !providers.is_empty() {
        sb = sb.with_execution_providers(providers)?;
    } else {
        log_warn("ONNX", "No GPU EP requested/available; using CPU EP");
    }

    // Build session
    let session: Session = sb.with_model_from_file(&args.model)?;
    log_ok("ONNX", &format!("model loaded: {}", args.model));

    // Kelas (cari license_plate utk overlay saja)
    let class_names: Vec<String> =
        serde_json::from_str(&fs::read_to_string(&args.classes).context("read classes.json")?)?;
    let plate_class_id = class_names
        .iter()
        .position(|s| normalize(s) == "licenseplate")
        .map(|i| i as i32);
    match plate_class_id {
        Some(id) => log_ok("CLASS", &format!("classes loaded ({}), plate_class_id={}", class_names.len(), id)),
        None => log_warn("CLASS", &format!("classes loaded ({}), plate_class_id=NONE", class_names.len())),
    }

    // MQTT off (kompatibel)
    log_warn("MQTT", "disabled for now (skip connect & publish)");
    let _touch_mqtt = (&args.mqtt_host, &args.mqtt_user, &args.mqtt_pass, &args.mqtt_client_id, &args.mqtt_topic);

    // MySQL (opsional)
    let db_pool: Option<Pool<MySql>> = if args.db_enable && !args.db_url.is_empty() {
        match MySqlPoolOptions::new().max_connections(5).connect(&args.db_url).await {
            Ok(pool) => {
                log_ok("DB", &format!("connected: {}", args.db_url));
                if let Err(e) = ensure_schema(&pool).await {
                    log_err("DB", &format!("ensure_schema: {:?}", e));
                } else {
                    log_ok("DB", "schema ready");
                }
                Some(pool)
            }
            Err(e) => { log_err("DB", &format!("connect error: {:?}", e)); None }
        }
    } else {
        log_warn("DB", "disabled or empty URL");
        None
    };

    // Kamera
    use videoio::{CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2};
    let mut cap = VideoCapture::new(args.cam, CAP_V4L2)
        .with_context(|| format!("Gagal buka /dev/video{}", args.cam))?;
    if !cap.is_opened()? {
        return Err(anyhow!("Device /dev/video{} tidak bisa dibuka", args.cam));
    }
    let _ = cap.set(CAP_PROP_FRAME_WIDTH, args.width as f64);
    let _ = cap.set(CAP_PROP_FRAME_HEIGHT, args.height as f64);
    let _ = cap.set(CAP_PROP_FPS, args.fps as f64);

    let fps_cam = cap.get(CAP_PROP_FPS)? as i32;
    let fps = if fps_cam > 0 { fps_cam } else { args.fps };
    let step = (fps / args.sample_fps).max(1);
    log_ok("CAM", &format!("open /dev/video{} {}x{}@{} (process every {} frame)", args.cam, args.width, args.height, fps, step));

    // State
    let mut total_vehicle: u32 = 0;
    let mut tracks: Vec<Track> = vec![];
    let mut next_id: u64 = 1;
    let mut frames_processed: u32 = 0;

    // Anti-duplikat (cache 1s)
    struct RecentCount { bbox: [f32; 4], center: (f32, f32), ts: Instant }
    let mut recent_counts: VecDeque<RecentCount> = VecDeque::new();
    let dup_window = Duration::from_millis(1000);

    // DB push periodik
    let mut last_db_push = Instant::now();
    const DB_PUSH_SECS: u64 = 10;
    let mut last_pushed_count: u32 = 0;

    // Garis referensi (opsional, tidak dipakai hitung)
    let parts: Vec<i32> = args.line.split(',').map(|s| s.trim().parse().unwrap_or(0)).collect();
    let (_p1, _p2) = ((parts[0] as f32, parts[1] as f32), (parts[2] as f32, parts[3] as f32));

    // Loop utama
    loop {
        // baca frame sesuai step
        let mut frame = Mat::default();
        for _ in 0..step {
            if !cap.read(&mut frame)? || frame.empty() {
                continue;
            }
        }
        if frame.empty() { continue; }

        // bersihkan cache duplikat lama
        while let Some(rc) = recent_counts.front() {
            if rc.ts.elapsed() > dup_window { recent_counts.pop_front(); } else { break; }
        }

        let orig_w = frame.cols();
        let orig_h = frame.rows();

        // preprocess + ONNX
        let (padded, info) = letterbox(&frame, args.imgsz)?;
        let input_chw: Array4<f32> = hwc_bgr_to_chw_rgb_normalized(&padded)?;
        let input_dyn = input_chw.into_dyn();
        let input_view: ArrayViewD<f32> = input_dyn.view();
        let input_cow: CowArray<'_, f32, IxDyn> = CowArray::from(input_view);
        let inputs = vec![Value::from_array(session.allocator(), &input_cow)?];

        let outputs = session.run(inputs)?;
        let out_tensor: ort::tensor::OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let out_array = out_tensor.view().to_owned().into_dyn();

        // decode YOLO
        let shape = out_array.shape().to_vec();
        let class_offset_is_second_dim = shape.len() == 3 && shape[1] < shape[2];

        let mut dets = decode_yolov8_output(
            &out_array, &info, orig_w, orig_h, class_offset_is_second_dim, args.conf_vehicle,
        );

        // pisah plate (untuk overlay label saja)
        let mut plates = Vec::<Detection>::new();
        if let Some(pid) = plate_class_id {
            let (mut veh, mut plc) = (Vec::new(), Vec::new());
            for d in dets.into_iter() {
                if d.class_id == pid {
                    if d.score >= args.conf_plate { plc.push(d); }
                } else {
                    veh.push(d);
                }
            }
            dets = veh;
            plates = plc;
        }

        // NMS
        let dets = nms(dets, args.nms_iou);
        let plates = nms(plates, 0.6);

        // tracking sederhana (IOU)
        let mut assigned = vec![false; dets.len()];
        for t in tracks.iter_mut() {
            let mut best = (None, 0.0f32);
            for (i, d) in dets.iter().enumerate() {
                if assigned[i] { continue; }
                let ov = iou(t.bbox_xyxy, d.bbox_xyxy);
                if ov > best.1 { best = (Some(i), ov); }
            }
            if let (Some(i), ov) = best {
                if ov > 0.3 {
                    t.bbox_xyxy = dets[i].bbox_xyxy;
                    t.miss = 0;
                    t.last_center = center(&t.bbox_xyxy);
                    t.age += 1;
                    assigned[i] = true;
                } else {
                    t.miss += 1;
                }
            } else {
                t.miss += 1;
            }
        }
        for (i, d) in dets.iter().enumerate() {
            if !assigned[i] {
                tracks.push(Track {
                    id: next_id, bbox_xyxy: d.bbox_xyxy, miss: 0, counted: false,
                    last_center: center(&d.bbox_xyxy), first_center: center(&d.bbox_xyxy), age: 1,
                });
                next_id += 1;
            }
        }
        tracks.retain(|t| t.miss <= 15);

        // counting anti-duplikat (age + movement + cache 1s)
        const MIN_AGE: u32 = 3;
        const MIN_MOVE: f32 = 30.0;
        const DUP_IOU: f32 = 0.5;
        const DUP_DIST: f32 = 60.0;

        for t in tracks.iter_mut() {
            t.last_center = center(&t.bbox_xyxy);
            if !t.counted && t.age >= MIN_AGE && dist(t.first_center, t.last_center) >= MIN_MOVE {
                let mut dup = false;
                for rc in recent_counts.iter() {
                    if iou(rc.bbox, t.bbox_xyxy) >= DUP_IOU || dist(rc.center, t.last_center) < DUP_DIST {
                        dup = true; break;
                    }
                }
                t.counted = true;
                if !dup {
                    total_vehicle += 1;
                    recent_counts.push_back(RecentCount { bbox: t.bbox_xyxy, center: t.last_center, ts: Instant::now() });
                }
            }
        }

        // PREVIEW
        if args.preview {
            let mut vis = frame.try_clone()?;

            // Gambar TRACK: hijau + label ID
            for t in &tracks {
                let x1 = t.bbox_xyxy[0].max(0.0) as i32;
                let y1 = t.bbox_xyxy[1].max(0.0) as i32;
                let x2 = t.bbox_xyxy[2].min((vis.cols()-1) as f32) as i32;
                let y2 = t.bbox_xyxy[3].min((vis.rows()-1) as f32) as i32;
                let w = (x2 - x1).max(1);
                let h = (y2 - y1).max(1);
                imgproc::rectangle(&mut vis, Rect::new(x1, y1, w, h), Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, 0)?;
                let label = format!("ID:{}", t.id);
                imgproc::put_text(&mut vis, &label,
                    opencv::core::Point::new(x1.max(0), (y1-6).max(16)),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar::new(255.0,255.0,255.0,0.0), 2, imgproc::LINE_AA, false)?;
            }

            // Gambar PLATE: kuning + label (overlay info saja)
            for p in &plates {
                let x1 = p.bbox_xyxy[0].max(0.0) as i32;
                let y1 = p.bbox_xyxy[1].max(0.0) as i32;
                let x2 = p.bbox_xyxy[2].min((vis.cols()-1) as f32) as i32;
                let y2 = p.bbox_xyxy[3].min((vis.rows()-1) as f32) as i32;
                let w = (x2 - x1).max(1);
                let h = (y2 - y1).max(1);
                imgproc::rectangle(&mut vis, Rect::new(x1, y1, w, h), Scalar::new(255.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, 0)?;
                imgproc::put_text(&mut vis, "PLATE",
                    opencv::core::Point::new(x1.max(0), (y1-6).max(16)),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar::new(255.0,255.0,0.0,0.0), 2, imgproc::LINE_AA, false)?;
            }

            // Ringkasan
            imgproc::put_text(&mut vis, &format!("VEH: {}", total_vehicle),
                opencv::core::Point::new(16, 32),
                imgproc::FONT_HERSHEY_SIMPLEX, 0.9, Scalar::new(255.0,255.0,255.0,0.0), 2, imgproc::LINE_AA, false)?;
            imgproc::put_text(&mut vis, &format!("frames: {}", frames_processed + 1),
                opencv::core::Point::new(16, 64),
                imgproc::FONT_HERSHEY_SIMPLEX, 0.8, Scalar::new(200.0,200.0,200.0,0.0), 2, imgproc::LINE_AA, false)?;
            show_preview(&vis);
        }

        frames_processed = frames_processed.saturating_add(1);

        // Push ke DB periodik (kolom 'count' kita isi vehicle total)
        if let Some(pool) = db_pool.as_ref() {
            if last_db_push.elapsed() >= Duration::from_secs(DB_PUSH_SECS) && total_vehicle != last_pushed_count {
                let now_ts = OffsetDateTime::now_utc().unix_timestamp();
                let msg = ClipCountMsg {
                    ts: now_ts,
                    cam_id: "cam01".into(),
                    clip_path: format!("rt-{}", now_ts),
                    count_non_ev: total_vehicle,
                    count_ev: 0,
                    frames_processed,
                    sample_fps: args.sample_fps,
                };
                match insert_clip_row(pool, &msg).await {
                    Ok(_) => { last_db_push = Instant::now(); last_pushed_count = total_vehicle; log_ok("DB", "inserted realtime snapshot"); }
                    Err(e) => log_err("DB", &format!("insert error: {:?}", e)),
                }
            }
        }
    }
}

/* ===================== DB ===================== */
async fn ensure_schema(pool: &Pool<MySql>) -> Result<()> {
    let sql = r#"
    CREATE TABLE IF NOT EXISTS clip_counts (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      ts TIMESTAMP NOT NULL,
      cam_id VARCHAR(64) NOT NULL,
      clip_path VARCHAR(512) NOT NULL UNIQUE,
      count INT NOT NULL,
      frames_processed INT NOT NULL,
      sample_fps INT NOT NULL,
      processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      INDEX (ts), INDEX (cam_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"#;
    sqlx::query(sql).execute(pool).await?;
    Ok(())
}
async fn insert_clip_row(pool: &Pool<MySql>, msg: &ClipCountMsg) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO clip_counts (ts, cam_id, clip_path, count, frames_processed, sample_fps)
        VALUES (FROM_UNIXTIME(?), ?, ?, ?, ?, ?)
        "#,
    )
    .bind(msg.ts)
    .bind(&msg.cam_id)
    .bind(&msg.clip_path)
    .bind(msg.count_non_ev as i64)
    .bind(msg.frames_processed as i64)
    .bind(msg.sample_fps)
    .execute(pool)
    .await?;
    Ok(())
}

/* ===================== UTIL ===================== */
#[derive(Clone, Debug)]
struct Detection { bbox_xyxy: [f32; 4], score: f32, class_id: i32 }
#[derive(Clone, Debug)]
struct Track {
    id: u64,
    bbox_xyxy: [f32; 4],
    miss: u32,
    counted: bool,
    last_center: (f32, f32),
    first_center: (f32, f32),
    age: u32,
}
fn center(b: &[f32; 4]) -> (f32, f32) { ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) }
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 { let dx = a.0 - b.0; let dy = a.1 - b.1; (dx*dx + dy*dy).sqrt() }
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let ix1 = ax1.max(bx1); let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2); let iy2 = ay2.min(by2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}
fn nms(mut dets: Vec<Detection>, iou_thr: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep = vec![]; let mut removed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if removed[i] { continue; }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if removed[j] { continue; }
            if iou(dets[i].bbox_xyxy, dets[j].bbox_xyxy) > iou_thr { removed[j] = true; }
        }
    }
    keep
}
struct LetterboxInfo { scale: f32, pad_x: f32, pad_y: f32}
fn letterbox(bgr: &Mat, new_size: i32) -> Result<(Mat, LetterboxInfo)> {
    let (h, w) = (bgr.rows(), bgr.cols());
    let r = (new_size as f32 / w as f32).min(new_size as f32 / h as f32);
    let new_w = (w as f32 * r).round() as i32;
    let new_h = (h as f32 * r).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(bgr, &mut resized, Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let dw = new_size - new_w; let dh = new_size - new_h;
    let top = dh / 2; let bottom = dh - top; let left = dw / 2; let right = dw - left;

    let mut padded = Mat::default();
    copy_make_border(&resized, &mut padded, top, bottom, left, right, BORDER_CONSTANT, Scalar::new(114.0,114.0,114.0,0.0))?;
    Ok((padded, LetterboxInfo { scale: r, pad_x: left as f32, pad_y: top as f32}))
}
fn hwc_bgr_to_chw_rgb_normalized(mat: &Mat) -> Result<Array4<f32>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    let mut f32m = Mat::default();
    rgb.convert_to(&mut f32m, CV_32F, 1.0 / 255.0, 0.0)?;

    let (h, w) = (f32m.rows() as usize, f32m.cols() as usize);
    let total = h * w * 3;
    let bytes = f32m.data_bytes()?;
    let mut hwc = vec![0f32; total];
    for i in 0..total {
        let p = &bytes[i * 4..i * 4 + 4];
        hwc[i] = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
    }
    let mut chw = vec![0f32; total];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let hwc_idx = (y * w + x) * 3 + ch;
                let chw_idx = ch * (h * w) + y * w + x;
                chw[chw_idx] = hwc[hwc_idx];
            }
        }
    }
    Ok(Array4::from_shape_vec((1, 3, h, w), chw)?)
}
fn decode_yolov8_output(
    output: &ndarray::ArrayD<f32>, info: &LetterboxInfo,
    orig_w: i32, orig_h: i32, class_offset_is_second_dim: bool, conf_thr: f32,
) -> Vec<Detection> {
    let view = output.view();
    let shape = view.shape().to_vec();
    let (n_dim, c_dim) = if class_offset_is_second_dim { (shape[2], shape[1]) } else { (shape[1], shape[2]) };

    let mut dets = Vec::<Detection>::new();
    for i in 0..n_dim {
        let read = |c: usize| -> f32 { if class_offset_is_second_dim { view[[0, c, i]] } else { view[[0, i, c]] } };
        let cx = read(0); let cy = read(1); let ww = read(2); let hh = read(3);

        // cari kelas score max
        let mut best_id = -1; let mut best_sc = 0.0f32;
        for c in 4..c_dim { let s = read(c); if s > best_sc { best_sc = s; best_id = (c - 4) as i32; } }
        if best_sc < conf_thr { continue; }

        // xywh(padded) -> xyxy(orig)
        let (mut x1, mut y1, mut x2, mut y2) = (cx - ww/2.0, cy - hh/2.0, cx + ww/2.0, cy + hh/2.0);
        x1 = (x1 - info.pad_x) / info.scale; y1 = (y1 - info.pad_y) / info.scale;
        x2 = (x2 - info.pad_x) / info.scale; y2 = (y2 - info.pad_y) / info.scale;
        x1 = x1.clamp(0.0, orig_w as f32 - 1.0); x2 = x2.clamp(0.0, orig_w as f32 - 1.0);
        y1 = y1.clamp(0.0, orig_h as f32 - 1.0); y2 = y2.clamp(0.0, orig_h as f32 - 1.0);

        dets.push(Detection { bbox_xyxy: [x1, y1, x2, y2], score: best_sc, class_id: best_id });
    }
    dets
}

/* ===================== (optional) EV helper - OFF ===================== */
#[allow(dead_code)]
fn is_ev_by_blue_strip(_bgr_plate: &Mat) -> Result<bool> { Ok(false) }
fn normalize(s: &str) -> String { s.to_ascii_lowercase().replace('_', "").replace(' ', "") }
