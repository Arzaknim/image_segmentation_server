#[derive(Debug, Clone)]
pub struct Tensor3d {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Tensor3d {
    pub fn zeros(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    pub fn from_vec(channels: usize, height: usize, width: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            channels * height * width,
            "Tensor3d data length does not match dimensions"
        );
        Self {
            channels,
            height,
            width,
            data,
        }
    }

    #[inline]
    pub fn index(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.height + y) * self.width + x
    }

    #[inline]
    pub fn get(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[self.index(c, y, x)]
    }

    #[inline]
    pub fn set(&mut self, c: usize, y: usize, x: usize, value: f32) {
        let idx = self.index(c, y, x);
        self.data[idx] = value;
    }

    pub fn channel_slice(&self, c: usize) -> &[f32] {
        let plane_size = self.height * self.width;
        let start = c * plane_size;
        &self.data[start..start + plane_size]
    }

    pub fn channel_slice_mut(&mut self, c: usize) -> &mut [f32] {
        let plane_size = self.height * self.width;
        let start = c * plane_size;
        &mut self.data[start..start + plane_size]
    }

    pub fn apply_swish_in_place(&mut self) {
        for val in &mut self.data {
            *val = swish(*val);
        }
    }

    pub fn apply_relu_in_place(&mut self) {
        for val in &mut self.data {
            *val = val.max(0.0);
        }
    }

    pub fn apply_sigmoid_in_place(&mut self) {
        for val in &mut self.data {
            *val = sigmoid(*val);
        }
    }

    pub fn add_in_place(&mut self, other: &Self) {
        assert_eq!(self.channels, other.channels);
        assert_eq!(self.height, other.height);
        assert_eq!(self.width, other.width);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            *dst += *src;
        }
    }

    pub fn scale_channels_in_place(&mut self, channel_scales: &Self) {
        assert_eq!(self.channels, channel_scales.channels);
        assert_eq!(channel_scales.height, 1);
        assert_eq!(channel_scales.width, 1);
        for c in 0..self.channels {
            let scale = channel_scales.data[c];
            for val in self.channel_slice_mut(c) {
                *val *= scale;
            }
        }
    }

    pub fn global_average_pool(&self) -> Self {
        let spatial_count = (self.height * self.width) as f32;
        let mut pooled = Vec::with_capacity(self.channels);
        for c in 0..self.channels {
            let sum: f32 = self.channel_slice(c).iter().copied().sum();
            pooled.push(sum / spatial_count);
        }
        Self::from_vec(self.channels, 1, 1, pooled)
    }

    pub fn upsample_nearest(&self, target_height: usize, target_width: usize) -> Self {
        let mut out = Self::zeros(self.channels, target_height, target_width);
        for c in 0..self.channels {
            let src_plane = self.channel_slice(c);
            let dst_plane = out.channel_slice_mut(c);
            for ty in 0..target_height {
                let sy = (ty * self.height) / target_height;
                let src_row = &src_plane[sy * self.width..(sy + 1) * self.width];
                let dst_row = &mut dst_plane[ty * target_width..(ty + 1) * target_width];
                for (tx, dst_val) in dst_row.iter_mut().enumerate() {
                    let sx = (tx * self.width) / target_width;
                    *dst_val = src_row[sx];
                }
            }
        }
        out
    }

    pub fn concat_channels(first: &Self, second: &Self) -> Self {
        assert_eq!(first.height, second.height);
        assert_eq!(first.width, second.width);
        let mut combined = Vec::with_capacity(first.data.len() + second.data.len());
        combined.extend_from_slice(&first.data);
        combined.extend_from_slice(&second.data);
        Self::from_vec(
            first.channels + second.channels,
            first.height,
            first.width,
            combined,
        )
    }

    pub fn softmax_along_channels(&self) -> Self {
        let mut out = Self::zeros(self.channels, self.height, self.width);
        let plane_size = self.height * self.width;

        for pixel_idx in 0..plane_size {
            let mut max_logit = f32::NEG_INFINITY;
            for c in 0..self.channels {
                let v = self.data[c * plane_size + pixel_idx];
                if v > max_logit {
                    max_logit = v;
                }
            }

            let mut exp_sum = 0.0f32;
            for c in 0..self.channels {
                let e = (self.data[c * plane_size + pixel_idx] - max_logit).exp();
                out.data[c * plane_size + pixel_idx] = e;
                exp_sum += e;
            }

            let inv_sum = 1.0 / exp_sum;
            for c in 0..self.channels {
                out.data[c * plane_size + pixel_idx] *= inv_sum;
            }
        }

        out
    }

    pub fn argmax_along_channels(&self) -> Vec<u8> {
        let plane_size = self.height * self.width;
        let mut mask = vec![0u8; plane_size];

        for (pixel_idx, out_class) in mask.iter_mut().enumerate() {
            let mut best_class = 0u8;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..self.channels {
                let score = self.data[c * plane_size + pixel_idx];
                if score > best_score {
                    best_score = score;
                    best_class = c as u8;
                }
            }
            *out_class = best_class;
        }

        mask
    }
}

#[inline]
pub fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Copy)]
pub struct Padding2d {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

impl Padding2d {
    pub const fn symmetric(pad: usize) -> Self {
        Self {
            left: pad,
            right: pad,
            top: pad,
            bottom: pad,
        }
    }

    pub const fn asymmetric(left: usize, right: usize, top: usize, bottom: usize) -> Self {
        Self {
            left,
            right,
            top,
            bottom,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    pub out_channels: usize,
    pub in_channels_per_group: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub groups: usize,
    pub padding: Padding2d,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Conv2dLayer {
    pub fn from_raw(
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        groups: usize,
        padding: Padding2d,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        assert_eq!(
            weights.len(),
            out_channels * in_channels_per_group * kernel_h * kernel_w
        );
        assert_eq!(bias.len(), out_channels);
        Self {
            out_channels,
            in_channels_per_group,
            kernel_h,
            kernel_w,
            stride,
            groups,
            padding,
            weights,
            bias,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_conv_and_batch_norm(
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        groups: usize,
        padding: Padding2d,
        conv_weights: &[f32],
        bn_weight: &[f32],
        bn_bias: &[f32],
        bn_running_mean: &[f32],
        bn_running_var: &[f32],
        bn_eps: f32,
    ) -> Self {
        let weights_per_filter = in_channels_per_group * kernel_h * kernel_w;
        let mut fused_weights = vec![0.0f32; conv_weights.len()];
        let mut fused_bias = vec![0.0f32; out_channels];

        for oc in 0..out_channels {
            let scale = bn_weight[oc] / (bn_running_var[oc] + bn_eps).sqrt();
            fused_bias[oc] = bn_bias[oc] - bn_running_mean[oc] * scale;

            let filter_start = oc * weights_per_filter;
            let src_filter = &conv_weights[filter_start..filter_start + weights_per_filter];
            let dst_filter = &mut fused_weights[filter_start..filter_start + weights_per_filter];
            for (dst, &src) in dst_filter.iter_mut().zip(src_filter.iter()) {
                *dst = src * scale;
            }
        }

        Self::from_raw(
            out_channels,
            in_channels_per_group,
            kernel_h,
            kernel_w,
            stride,
            groups,
            padding,
            fused_weights,
            fused_bias,
        )
    }

    pub fn forward(&self, input: &Tensor3d) -> Tensor3d {
        let expected_in_channels = self.in_channels_per_group * self.groups;
        assert_eq!(input.channels, expected_in_channels);

        let padded_h = input.height + self.padding.top + self.padding.bottom;
        let padded_w = input.width + self.padding.left + self.padding.right;
        let out_h = (padded_h - self.kernel_h) / self.stride + 1;
        let out_w = (padded_w - self.kernel_w) / self.stride + 1;

        if self.kernel_h == 1
            && self.kernel_w == 1
            && self.stride == 1
            && self.groups == 1
            && self.padding.left == 0
            && self.padding.top == 0
        {
            return self.forward_pointwise(input, out_h, out_w);
        }

        if self.groups == input.channels && self.in_channels_per_group == 1 {
            return self.forward_depthwise(input, out_h, out_w);
        }

        self.forward_general(input, out_h, out_w)
    }

    fn forward_pointwise(&self, input: &Tensor3d, out_h: usize, out_w: usize) -> Tensor3d {
        let mut output = Tensor3d::zeros(self.out_channels, out_h, out_w);
        let plane_size = out_h * out_w;

        for oc in 0..self.out_channels {
            let bias_val = self.bias[oc];
            let out_plane = output.channel_slice_mut(oc);
            out_plane.fill(bias_val);

            let weight_row = &self.weights[oc * self.in_channels_per_group..(oc + 1) * self.in_channels_per_group];
            for (ic, &w) in weight_row.iter().enumerate() {
                let in_plane = &input.data[ic * plane_size..(ic + 1) * plane_size];
                for (dst, &src) in out_plane.iter_mut().zip(in_plane.iter()) {
                    *dst += src * w;
                }
            }
        }

        output
    }

    fn forward_depthwise(&self, input: &Tensor3d, out_h: usize, out_w: usize) -> Tensor3d {
        let mut output = Tensor3d::zeros(self.out_channels, out_h, out_w);
        let kernel_area = self.kernel_h * self.kernel_w;

        for c in 0..self.out_channels {
            let bias_val = self.bias[c];
            let kernel = &self.weights[c * kernel_area..(c + 1) * kernel_area];
            let in_plane = input.channel_slice(c);
            let out_plane = output.channel_slice_mut(c);

            for oy in 0..out_h {
                let base_iy = oy as isize * self.stride as isize - self.padding.top as isize;
                for ox in 0..out_w {
                    let base_ix = ox as isize * self.stride as isize - self.padding.left as isize;
                    let mut acc = bias_val;

                    for ky in 0..self.kernel_h {
                        let iy = base_iy + ky as isize;
                        if iy < 0 || iy >= input.height as isize {
                            continue;
                        }
                        let in_row_offset = iy as usize * input.width;
                        let k_row_offset = ky * self.kernel_w;

                        for kx in 0..self.kernel_w {
                            let ix = base_ix + kx as isize;
                            if ix >= 0 && ix < input.width as isize {
                                acc += in_plane[in_row_offset + ix as usize]
                                    * kernel[k_row_offset + kx];
                            }
                        }
                    }

                    out_plane[oy * out_w + ox] = acc;
                }
            }
        }

        output
    }

    fn forward_general(&self, input: &Tensor3d, out_h: usize, out_w: usize) -> Tensor3d {
        let mut output = Tensor3d::zeros(self.out_channels, out_h, out_w);
        let kernel_area = self.kernel_h * self.kernel_w;
        let out_channels_per_group = self.out_channels / self.groups;

        for oc in 0..self.out_channels {
            let group_idx = oc / out_channels_per_group;
            let in_channel_start = group_idx * self.in_channels_per_group;
            let out_plane = output.channel_slice_mut(oc);
            out_plane.fill(self.bias[oc]);

            for ic_local in 0..self.in_channels_per_group {
                let ic = in_channel_start + ic_local;
                let in_plane = input.channel_slice(ic);
                let weight_offset = (oc * self.in_channels_per_group + ic_local) * kernel_area;
                let kernel = &self.weights[weight_offset..weight_offset + kernel_area];

                for oy in 0..out_h {
                    let base_iy = oy as isize * self.stride as isize - self.padding.top as isize;
                    let out_row = &mut out_plane[oy * out_w..(oy + 1) * out_w];

                    for ky in 0..self.kernel_h {
                        let iy = base_iy + ky as isize;
                        if iy < 0 || iy >= input.height as isize {
                            continue;
                        }
                        let in_row = &in_plane[iy as usize * input.width..(iy as usize + 1) * input.width];
                        let k_row = &kernel[ky * self.kernel_w..(ky + 1) * self.kernel_w];

                        for (ox, dst) in out_row.iter_mut().enumerate() {
                            let base_ix = ox as isize * self.stride as isize - self.padding.left as isize;
                            for (kx, &k_val) in k_row.iter().enumerate() {
                                let ix = base_ix + kx as isize;
                                if ix >= 0 && ix < input.width as isize {
                                    *dst += in_row[ix as usize] * k_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        output
    }
}

