use anyhow::Result;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor, vision::dataset::Dataset};

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    opt: nn::Optimizer<nn::Adam>
}

impl Net {
    pub fn new(vs: &nn::VarStore) -> Net {
        let mut cfg = nn::ConvConfig::default();
        cfg.padding = 1;
        let conv1 = nn::conv2d(vs.root(), 1, 32, 2, cfg);
        let conv2 = nn::conv2d(vs.root(), 32, 64, 3, Default::default());
        let fc1 = nn::linear(vs.root(), 256, 256, Default::default());
        let fc2 = nn::linear(vs.root(), 256, 1, Default::default());
        let opt = nn::Adam::default().build(vs, 1e-4).unwrap();
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
            opt,
        }
    }

    pub fn train(&mut self, dataset: &Dataset) {
        for _ in 1..50 {
            for (bimages, blabels) in dataset.train_iter(256).shuffle().to_device(Device::cuda_if_available()) {
                let ans = self.forward_t(&bimages, true);
                let loss = (ans - blabels).norm();
                self.opt.backward_step(&loss);
            }
        }
    }

    pub fn eval(&self, t: Tensor) -> f32 {
        let ans = self.forward_t(&t, false);
        ans.reshape(&[1]).double_value(&[0]) as f32
    }
}

impl ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 3, 3])
            .apply(&self.conv1)
            .avg_pool2d(&[2, 2], &[1, 1], &[1, 1], false, true, 4)
            .apply(&self.conv2)
            .avg_pool2d(&[2, 2], &[1, 1], &[0, 0], false, false, 4)
            .view([-1, 256])
            .apply(&self.fc1)
            .relu()
            .dropout_(0.5, train)
            .apply(&self.fc2)
            .tanh()
    }
}