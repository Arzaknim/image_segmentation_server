use std::path::Path;

use crate::pth_loader::{PthError, PthStateDict};
use crate::tensor::{Conv2dLayer, Padding2d, Tensor3d};

const ENCODER_BN_EPS: f32 = 1e-3;
const DECODER_BN_EPS: f32 = 1e-5;
const NUM_CLASSES: usize = 6;
const ENCODER_OUT_INDEXES: [usize; 4] = [2, 4, 8, 15];

#[derive(Debug, Clone, Copy)]
struct MbConvConfig {
    block_index: usize,
    in_channels: usize,
    expanded_channels: usize,
    squeezed_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    has_expansion: bool,
    has_skip: bool,
}

const EFFICIENTNET_B0_BLOCKS: [MbConvConfig; 16] = [
    MbConvConfig { block_index: 0,  in_channels: 32,  expanded_channels: 32,   squeezed_channels: 8,  out_channels: 16,  kernel_size: 3, stride: 1, has_expansion: false, has_skip: false },
    MbConvConfig { block_index: 1,  in_channels: 16,  expanded_channels: 96,   squeezed_channels: 4,  out_channels: 24,  kernel_size: 3, stride: 2, has_expansion: true,  has_skip: false },
    MbConvConfig { block_index: 2,  in_channels: 24,  expanded_channels: 144,  squeezed_channels: 6,  out_channels: 24,  kernel_size: 3, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 3,  in_channels: 24,  expanded_channels: 144,  squeezed_channels: 6,  out_channels: 40,  kernel_size: 5, stride: 2, has_expansion: true,  has_skip: false },
    MbConvConfig { block_index: 4,  in_channels: 40,  expanded_channels: 240,  squeezed_channels: 10, out_channels: 40,  kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 5,  in_channels: 40,  expanded_channels: 240,  squeezed_channels: 10, out_channels: 80,  kernel_size: 3, stride: 2, has_expansion: true,  has_skip: false },
    MbConvConfig { block_index: 6,  in_channels: 80,  expanded_channels: 480,  squeezed_channels: 20, out_channels: 80,  kernel_size: 3, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 7,  in_channels: 80,  expanded_channels: 480,  squeezed_channels: 20, out_channels: 80,  kernel_size: 3, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 8,  in_channels: 80,  expanded_channels: 480,  squeezed_channels: 20, out_channels: 112, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: false },
    MbConvConfig { block_index: 9,  in_channels: 112, expanded_channels: 672,  squeezed_channels: 28, out_channels: 112, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 10, in_channels: 112, expanded_channels: 672,  squeezed_channels: 28, out_channels: 112, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 11, in_channels: 112, expanded_channels: 672,  squeezed_channels: 28, out_channels: 192, kernel_size: 5, stride: 2, has_expansion: true,  has_skip: false },
    MbConvConfig { block_index: 12, in_channels: 192, expanded_channels: 1152, squeezed_channels: 48, out_channels: 192, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 13, in_channels: 192, expanded_channels: 1152, squeezed_channels: 48, out_channels: 192, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 14, in_channels: 192, expanded_channels: 1152, squeezed_channels: 48, out_channels: 192, kernel_size: 5, stride: 1, has_expansion: true,  has_skip: true  },
    MbConvConfig { block_index: 15, in_channels: 192, expanded_channels: 1152, squeezed_channels: 48, out_channels: 320, kernel_size: 3, stride: 1, has_expansion: true,  has_skip: false },
];

#[derive(Debug, Clone)]
pub struct SqueezeExcitation {
    reduce: Conv2dLayer,
    expand: Conv2dLayer,
}

impl SqueezeExcitation {
    pub fn forward(&self, input: &mut Tensor3d) {
        let pooled = input.global_average_pool();
        let mut squeezed = self.reduce.forward(&pooled);
        squeezed.apply_swish_in_place();
        let mut gates = self.expand.forward(&squeezed);
        gates.apply_sigmoid_in_place();
        input.scale_channels_in_place(&gates);
    }
}

#[derive(Debug, Clone)]
pub struct MbConvBlock {
    expand_conv: Option<Conv2dLayer>,
    depthwise_conv: Conv2dLayer,
    se: SqueezeExcitation,
    project_conv: Conv2dLayer,
    has_skip: bool,
}

impl MbConvBlock {
    pub fn forward(&self, input: &Tensor3d) -> Tensor3d {
        let mut x = if let Some(expand) = &self.expand_conv {
            let mut expanded = expand.forward(input);
            expanded.apply_swish_in_place();
            expanded
        } else {
            input.clone()
        };

        x = self.depthwise_conv.forward(&x);
        x.apply_swish_in_place();
        self.se.forward(&mut x);
        x = self.project_conv.forward(&x);

        if self.has_skip {
            x.add_in_place(input);
        }

        x
    }
}

#[derive(Debug, Clone)]
pub struct EfficientNetB0Encoder {
    conv_stem: Conv2dLayer,
    blocks: Vec<MbConvBlock>,
}

impl EfficientNetB0Encoder {
    pub fn forward(&self, input: &Tensor3d) -> Vec<Tensor3d> {
        let mut features = Vec::with_capacity(6);
        features.push(input.clone());

        let mut x = self.conv_stem.forward(input);
        x.apply_swish_in_place();
        features.push(x.clone());

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x);
            if ENCODER_OUT_INDEXES.contains(&block_idx) {
                features.push(x.clone());
            }
        }

        features
    }
}

#[derive(Debug, Clone)]
pub struct UnetDecoderBlock {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl UnetDecoderBlock {
    pub fn forward(
        &self,
        feature_map: &Tensor3d,
        target_height: usize,
        target_width: usize,
        skip_connection: Option<&Tensor3d>,
    ) -> Tensor3d {
        let upsampled = feature_map.upsample_nearest(target_height, target_width);
        let merged = match skip_connection {
            Some(skip) => Tensor3d::concat_channels(&upsampled, skip),
            None => upsampled,
        };

        let mut x = self.conv1.forward(&merged);
        x.apply_relu_in_place();
        x = self.conv2.forward(&x);
        x.apply_relu_in_place();
        x
    }
}

#[derive(Debug, Clone)]
pub struct EfficientNetB0Unet {
    encoder: EfficientNetB0Encoder,
    decoder_blocks: Vec<UnetDecoderBlock>,
    segmentation_head: Conv2dLayer,
}

impl EfficientNetB0Unet {
    pub fn from_pth_file<P: AsRef<Path>>(path: P) -> Result<Self, PthError> {
        let state_dict = PthStateDict::load_from_file(path)?;
        Self::from_state_dict(&state_dict)
    }

    pub fn from_state_dict(state_dict: &PthStateDict) -> Result<Self, PthError> {
        let conv_stem = load_fused_conv_bn(
            state_dict,
            "encoder._conv_stem.weight",
            "encoder._bn0",
            32,
            3,
            3,
            2,
            1,
            static_same_padding(3, 2),
            ENCODER_BN_EPS,
        )?;

        let mut blocks = Vec::with_capacity(EFFICIENTNET_B0_BLOCKS.len());
        for cfg in EFFICIENTNET_B0_BLOCKS {
            blocks.push(load_mbconv_block(state_dict, cfg)?);
        }

        let decoder_specs = [
            (0, 432, 256),
            (1, 296, 128),
            (2, 152, 64),
            (3, 96, 32),
            (4, 32, 16),
        ];
        let mut decoder_blocks = Vec::with_capacity(decoder_specs.len());
        for (idx, in_ch, out_ch) in decoder_specs {
            decoder_blocks.push(load_decoder_block(state_dict, idx, in_ch, out_ch)?);
        }

        let head_weights = state_dict
            .get_tensor("segmentation_head.0.weight", &[NUM_CLASSES, 16, 3, 3])?
            .to_vec();
        let head_bias = state_dict
            .get_tensor("segmentation_head.0.bias", &[NUM_CLASSES])?
            .to_vec();
        let segmentation_head = Conv2dLayer::from_raw(
            NUM_CLASSES,
            16,
            3,
            3,
            1,
            1,
            Padding2d::symmetric(1),
            head_weights,
            head_bias,
        );

        Ok(Self {
            encoder: EfficientNetB0Encoder { conv_stem, blocks },
            decoder_blocks,
            segmentation_head,
        })
    }

    pub fn forward(&self, input: &Tensor3d) -> Tensor3d {
        let features = self.encoder.forward(input);
        let spatial_shapes: Vec<(usize, usize)> = features
            .iter()
            .rev()
            .map(|t| (t.height, t.width))
            .collect();

        let reversed_features: Vec<&Tensor3d> = features[1..].iter().rev().collect();
        let mut x = reversed_features[0].clone();
        let skip_connections = &reversed_features[1..];

        for (i, decoder_block) in self.decoder_blocks.iter().enumerate() {
            let (target_h, target_w) = spatial_shapes[i + 1];
            let skip = skip_connections.get(i).copied();
            x = decoder_block.forward(&x, target_h, target_w, skip);
        }

        self.segmentation_head.forward(&x)
    }
}

fn static_same_padding(kernel_size: usize, stride: usize) -> Padding2d {
    match (kernel_size, stride) {
        (1, 1) => Padding2d::symmetric(0),
        (3, 1) => Padding2d::symmetric(1),
        (5, 1) => Padding2d::symmetric(2),
        (3, 2) => Padding2d::asymmetric(0, 1, 0, 1),
        (5, 2) => Padding2d::asymmetric(1, 2, 1, 2),
        _ => Padding2d::symmetric(kernel_size / 2),
    }
}

fn load_mbconv_block(state_dict: &PthStateDict, cfg: MbConvConfig) -> Result<MbConvBlock, PthError> {
    let prefix = format!("encoder._blocks.{}", cfg.block_index);

    let expand_conv = if cfg.has_expansion {
        Some(load_fused_conv_bn(
            state_dict,
            &format!("{prefix}._expand_conv.weight"),
            &format!("{prefix}._bn0"),
            cfg.expanded_channels,
            cfg.in_channels,
            1,
            1,
            1,
            Padding2d::symmetric(0),
            ENCODER_BN_EPS,
        )?)
    } else {
        None
    };

    let depthwise_conv = load_fused_conv_bn(
        state_dict,
        &format!("{prefix}._depthwise_conv.weight"),
        &format!("{prefix}._bn1"),
        cfg.expanded_channels,
        1,
        cfg.kernel_size,
        cfg.stride,
        cfg.expanded_channels,
        static_same_padding(cfg.kernel_size, cfg.stride),
        ENCODER_BN_EPS,
    )?;

    let se_reduce_w = state_dict
        .get_tensor(
            &format!("{prefix}._se_reduce.weight"),
            &[cfg.squeezed_channels, cfg.expanded_channels, 1, 1],
        )?
        .to_vec();
    let se_reduce_b = state_dict
        .get_tensor(&format!("{prefix}._se_reduce.bias"), &[cfg.squeezed_channels])?
        .to_vec();
    let se_expand_w = state_dict
        .get_tensor(
            &format!("{prefix}._se_expand.weight"),
            &[cfg.expanded_channels, cfg.squeezed_channels, 1, 1],
        )?
        .to_vec();
    let se_expand_b = state_dict
        .get_tensor(&format!("{prefix}._se_expand.bias"), &[cfg.expanded_channels])?
        .to_vec();

    let se = SqueezeExcitation {
        reduce: Conv2dLayer::from_raw(
            cfg.squeezed_channels,
            cfg.expanded_channels,
            1,
            1,
            1,
            1,
            Padding2d::symmetric(0),
            se_reduce_w,
            se_reduce_b,
        ),
        expand: Conv2dLayer::from_raw(
            cfg.expanded_channels,
            cfg.squeezed_channels,
            1,
            1,
            1,
            1,
            Padding2d::symmetric(0),
            se_expand_w,
            se_expand_b,
        ),
    };

    let project_conv = load_fused_conv_bn(
        state_dict,
        &format!("{prefix}._project_conv.weight"),
        &format!("{prefix}._bn2"),
        cfg.out_channels,
        cfg.expanded_channels,
        1,
        1,
        1,
        Padding2d::symmetric(0),
        ENCODER_BN_EPS,
    )?;

    Ok(MbConvBlock {
        expand_conv,
        depthwise_conv,
        se,
        project_conv,
        has_skip: cfg.has_skip,
    })
}

fn load_decoder_block(
    state_dict: &PthStateDict,
    block_idx: usize,
    in_channels: usize,
    out_channels: usize,
) -> Result<UnetDecoderBlock, PthError> {
    let prefix = format!("decoder.blocks.{block_idx}");
    let conv1 = load_fused_conv_bn(
        state_dict,
        &format!("{prefix}.conv1.0.weight"),
        &format!("{prefix}.conv1.1"),
        out_channels,
        in_channels,
        3,
        1,
        1,
        Padding2d::symmetric(1),
        DECODER_BN_EPS,
    )?;
    let conv2 = load_fused_conv_bn(
        state_dict,
        &format!("{prefix}.conv2.0.weight"),
        &format!("{prefix}.conv2.1"),
        out_channels,
        out_channels,
        3,
        1,
        1,
        Padding2d::symmetric(1),
        DECODER_BN_EPS,
    )?;
    Ok(UnetDecoderBlock { conv1, conv2 })
}

#[allow(clippy::too_many_arguments)]
fn load_fused_conv_bn(
    state_dict: &PthStateDict,
    conv_weight_name: &str,
    bn_prefix: &str,
    out_channels: usize,
    in_channels_per_group: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    padding: Padding2d,
    bn_eps: f32,
) -> Result<Conv2dLayer, PthError> {
    let conv_weights = state_dict.get_tensor(
        conv_weight_name,
        &[out_channels, in_channels_per_group, kernel_size, kernel_size],
    )?;
    let bn_weight = state_dict.get_tensor(&format!("{bn_prefix}.weight"), &[out_channels])?;
    let bn_bias = state_dict.get_tensor(&format!("{bn_prefix}.bias"), &[out_channels])?;
    let bn_mean = state_dict.get_tensor(&format!("{bn_prefix}.running_mean"), &[out_channels])?;
    let bn_var = state_dict.get_tensor(&format!("{bn_prefix}.running_var"), &[out_channels])?;

    Ok(Conv2dLayer::from_conv_and_batch_norm(
        out_channels,
        in_channels_per_group,
        kernel_size,
        kernel_size,
        stride,
        groups,
        padding,
        conv_weights,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        bn_eps,
    ))
}

