use std::error::Error;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::inference::SegmentationPredictor;
use crate::tensor::Tensor3d;

static ORT_ENVIRONMENT: OnceLock<Environment> = OnceLock::new();

fn get_or_init_environment() -> Result<&'static Environment, Box<dyn Error>> {
    let env = ORT_ENVIRONMENT.get_or_init(|| {
        ort::init()
            .with_name("dubai_segmentation_ort")
            .build()
            .expect("Failed to initialize ONNX Runtime environment")
    });
    Ok(env)
}

pub struct OnnxModel {
    session: Mutex<Session>,
}

impl OnnxModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn Error>> {
        let path_ref = model_path.as_ref();
        if !path_ref.exists() {
            return Err(format!("ONNX model file not found at '{}'", path_ref.display()).into());
        }

        let environment = get_or_init_environment()?;

        let mut builder = Session::builder(environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;
        let session = builder.commit_from_file(path_ref)?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl SegmentationPredictor for OnnxModel {
    fn predict_tile(&self, tile: &Tensor3d) -> Result<Tensor3d, Box<dyn Error>> {
        let input_tensor = Tensor::from_array((
            [1usize, tile.channels, tile.height, tile.width],
            tile.data.clone().into_boxed_slice(),
        ))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("Failed to acquire ONNX session lock: {e}"))?;

        let outputs = session.run(ort::inputs![input_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 4 {
            return Err(format!(
                "Expected 4D model output [B, C, H, W], got shape {shape:?}"
            )
            .into());
        }

        let num_classes = shape[1] as usize;
        let height = shape[2] as usize;
        let width = shape[3] as usize;

        let logits: Vec<f32> = data.to_vec();
        Ok(Tensor3d::from_vec(num_classes, height, width, logits))
    }
}

