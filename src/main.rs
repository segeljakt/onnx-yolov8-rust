use image::imageops::FilterType;
use image::GenericImageView;
use ndarray::s;
use ndarray::Array;
use ndarray::Axis;
use ndarray::CowArray;
use ndarray::IxDyn;
use ort::Environment;
use ort::ExecutionProvider;
use ort::GraphOptimizationLevel;
use ort::SessionBuilder;
use ort::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read("cats.jpg")?;
    let (input, img_width, img_height) = prepare_input(data);
    let output = run_model(input);
    let boxes = process_output(output, img_width, img_height);
    for (x1, y1, x2, y2, label, prob) in boxes {
        println!(
            "x1: {}, y1: {}, x2: {}, y2: {}, label: {}, prob: {}",
            x1, y1, x2, y2, label, prob
        );
    }
    Ok(())
}

fn prepare_input(buf: Vec<u8>) -> (Array<f32, IxDyn>, u32, u32) {
    let img = image::load_from_memory(&buf).unwrap();
    let (img_width, img_height) = (img.width(), img.height());
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    for (x, y, rgb) in img.pixels() {
        let x = x as usize;
        let y = y as usize;
        let [r, g, b, _] = rgb.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    }
    (input, img_width, img_height)
}

fn run_model(input: Array<f32, IxDyn>) -> Array<f32, IxDyn> {
    let env = Environment::builder()
        .with_name("YOLOv8")
        // .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .with_execution_providers([ExecutionProvider::CoreML(Default::default())])
        .build()
        .unwrap()
        .into_arc();
    let t0 = std::time::Instant::now();
    let session = SessionBuilder::new(&env)
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_model_from_file("yolov8m.onnx")
        .unwrap();
    let t1 = std::time::Instant::now();
    println!("Model loading time: {:?}", t1 - t0);
    let input = CowArray::from(input);
    let x = Value::from_array(session.allocator(), &input).unwrap();
    let t0 = std::time::Instant::now();
    let y = session.run(vec![x]).unwrap();
    let t1 = std::time::Instant::now();
    println!("Inference time: {:?}", t1 - t0);
    y[0].try_extract::<f32>().unwrap().view().t().into_owned()
}

fn process_output(
    output: Array<f32, IxDyn>,
    img_width: u32,
    img_height: u32,
) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.5 {
            continue;
        }
        let label = YOLO_CLASSES[class_id];
        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;
        boxes.push((x1, y1, x2, y2, label, prob));
    }

    boxes.sort_by(|box1, box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len() > 0 {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| iou(&boxes[0], box1) < 0.7)
            .map(|x| *x)
            .collect()
    }
    return result;
}

fn iou(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

fn union(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    let box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

fn intersection(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2 - x1) * (y2 - y1);
}

const YOLO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
