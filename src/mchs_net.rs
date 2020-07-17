use std::collections::{HashMap};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use tch::{Tensor, vision::dataset::Dataset, Device};

use crate::game::*;
use crate::net::Net;
use crate::mchs::{uct};

pub const ROOT: i32 = 611669;

pub fn code_to_tensor(code: i32) -> Tensor {
    Tensor::of_slice(&(0..9).map(|i| ((code >> 2 * i) % 4 - 1) as f32).collect::<Vec<f32>>()).to_device(Device::cuda_if_available())
}

pub fn eval_net(code: i32, net: &Net) -> f32 {
    net.eval(code_to_tensor(code))
}

pub struct MCHSNet {
    // code, (parents, children, w, n)
    pub h: HashMap<i32, (Vec<i32>, Vec<i32>, f32, f32)>,
    pub turn: i8,
    pub hist: Vec<i32>,
    pub ud: Uniform<f32>,
}

impl MCHSNet {
    pub fn new(turn: i8) -> Self {
        let mut h = HashMap::new();
        h.insert(ROOT, (Vec::new(), Vec::new(), 0.0, 1.0));
        let hist = Vec::new();
        let ud = Uniform::new(0.0, 1.0);
        Self { h, turn, hist, ud }
    }

    pub fn init(&mut self, turn: i8) {
        self.h = HashMap::new();
        self.h.insert(ROOT, (Vec::new(), Vec::new(), 0.0, 1.0));
        self.turn = turn;
        self.hist = Vec::new();
    }

    pub fn new_node(&mut self, code: i32) {
        self.h.entry(code).or_insert((Vec::new(), Vec::new(), 0.0, 1.0));
    }

    pub fn receive(&mut self, code: i32) {
        self.hist.push(code);
        self.new_node(code);
    }

    pub fn expand(&mut self, code: i32, net: &Net) -> (f32, f32) {
        let mut dn = 0.0;
        let mut dw = 0.0;
        let w = winner(code);
        if w != 2 {
            dw = w as f32;
            dn = 1.0;
        } else {
            for mov in 0..9 {
                if valid(code, mov) {
                    let next = next_code(code, mov);
                    self.h.get_mut(&code).unwrap().1.push(next);
                    self.new_node(next);
                    self.h.get_mut(&next).unwrap().0.push(code);
                    let w = eval_net(next, net);
                    self.h.get_mut(&next).unwrap().2 += w;
                    dn += 1.0;
                    dw += w;
                }
            }
        }
        self.h.get_mut(&code).unwrap().2 += dw;
        self.h.get_mut(&code).unwrap().3 += dn;
        (dw, dn)
    }

    pub fn search(&mut self, root: i32, n_iter: usize, net: &Net, train: bool) -> usize {
        for _ in 0..n_iter {
            let (w, n) = self.search_one(root, net);
            self.h.get_mut(&root).unwrap().2 += w;
            self.h.get_mut(&root).unwrap().3 += n;
        }
        let choice = self.choice_from_pi(root, train);
        let next = self.h[&root].1[choice];
        self.hist.push(next);
        move_diff(root, next)
    }

    pub fn search_one(&mut self, code: i32, net: &Net) -> (f32, f32) {
        if self.h[&code].1.len() == 0 {
            self.expand(code, net)
        } else {
            let choice = self.choice_from_uct(code, true);
            let (dw, dn) = self.search_one(self.h[&code].1[choice], net);
            self.h.get_mut(&code).unwrap().2 += dw;
            self.h.get_mut(&code).unwrap().3 += dn;
            (dw, dn)
        }
    }

    pub fn choice_from_uct(&self, code: i32, c_flg: bool) -> usize {
        let np = self.h[&code].3;
        let turn = turn(code);
        let v_uct = self.h[&code].1.iter()
                                   .map(|co| {
                                       let w = self.h[co].2;
                                       let ni = self.h[co].3;
                                       uct(turn, w as f64, np as f64, ni as f64, c_flg) as f32
                                   }).collect::<Vec<f32>>();
        let mut choice = 0;
        let mut u_max = std::f32::MIN;
        for (i, &u) in v_uct.iter().enumerate() {
            if u > u_max {
                choice = i;
                u_max = u;
            }
        }
        choice
    }

    pub fn choice_from_pi(&self, code: i32, tau: bool) -> usize {
        let mut n_sum = 0.0;
        let mut n_max = 0.0;
        let mut choice = 0;
        for (i, co) in self.h[&code].1.iter().enumerate() {
            let ni = self.h[co].3;
            n_sum += ni;
            if ni > n_max {
                choice = i;
                n_max = ni;
            }
        }
        if tau {
            let pi = self.h[&code].1.iter()
                                    .map(|co| self.h[co].3 / n_sum)
                                    .collect::<Vec<f32>>();
            let mut r = self.ud.sample(&mut thread_rng());
            let mut count = 0;
            while r > 0.0 {
                r -= pi[count];
                count += 1;
            }
            choice = count - 1;
        }
        choice
    }
}

pub struct SaveData {
    pub codes: Vec<i32>,
    pub results: Vec<i8>
}

impl SaveData {
    pub fn new() -> Self {
        let codes = Vec::new();
        let results = Vec::new();
        Self { codes, results }
    }

    pub fn save(&mut self, mchs_net: &MCHSNet, result: i8) {
        for &c in mchs_net.hist.iter() {
            self.codes.push(c);
            self.results.push(result);
        }
    }

    pub fn init(&mut self) {
        self.codes = Vec::new();
        self.results = Vec::new();
    }

    pub fn to_dataset(&self) -> Dataset {
        let l = self.codes.len();
        let mut img_v = Vec::with_capacity(l);
        let mut lab_v = Vec::with_capacity(l);
        for (&c, &r) in self.codes.iter().zip(self.results.iter()) {
            for i in 0..9 {
                img_v.push(((c >> i * 2) % 4 - 1) as f32);
            }
            lab_v.push(r);
        }
        let train_images = Tensor::of_slice(&img_v).reshape(&[-1, 9]);
        let train_labels = Tensor::of_slice(&lab_v);
        let test_images = Tensor::empty(&[1, 9], (tch::Kind::Float, Device::Cpu));
        let test_labels = Tensor::empty(&[1], (tch::Kind::Float, Device::Cpu));
        let labels = 3;
        Dataset { train_images, train_labels, test_images, test_labels, labels }
    }
}