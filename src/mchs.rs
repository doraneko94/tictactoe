use std::collections::{HashMap};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

use crate::game::*;

pub const ROOT: i32 = 611669;

pub fn uct(turn: i8, w: f64, np: f64, ni: f64, c_flg: bool) -> f64 {
    let c = if c_flg {
        ((np + 19653.0) / 19652.0).ln() + 1.25
    } else {
        0.0
    };
    turn as f64 * w / ni + c * (np.ln() / ni).sqrt()
}

pub fn eval(code: i32) -> i8 {
    let mut c = code;
    while winner(c) == 2 {
        let mut cand = Vec::new();
        for i in 0..9 {
            if valid(c, i) {
                cand.push(next_code(c, i));
            }
        }
        let ud = Uniform::new(0usize, cand.len());
        c = cand[ud.sample(&mut thread_rng())];
    }
    winner(c)
}

pub struct MCHS {
    // code, (parents, children, w, n)
    pub h: HashMap<i32, (Vec<i32>, Vec<i32>, f64, f64)>,
    pub turn: i8,
}

impl MCHS {
    pub fn new(turn: i8) -> Self {
        let mut h = HashMap::new();
        h.insert(ROOT, (Vec::new(), Vec::new(), 0.0, 1.0));
        Self { h, turn }
    }

    pub fn new_node(&mut self, code: i32) {
        self.h.entry(code).or_insert((Vec::new(), Vec::new(), 0.0, 1.0));
    }

    pub fn expand(&mut self, code: i32) -> (f64, f64) {
        let mut dn = 0.0;
        let mut dw = 0.0;
        let w = winner(code);
        if w != 2 {
            dw = w as f64;
            dn = 1.0;
        } else {
            for mov in 0..9 {
                if valid(code, mov) {
                    let next = next_code(code, mov);
                    self.h.get_mut(&code).unwrap().1.push(next);
                    self.new_node(next);
                    self.h.get_mut(&next).unwrap().0.push(code);
                    let w = eval(next) as f64;
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

    pub fn search(&mut self, root: i32, n_iter: usize) -> usize {
        for _ in 0..n_iter {
            let (w, n) = self.search_one(root);
            self.h.get_mut(&root).unwrap().2 += w;
            self.h.get_mut(&root).unwrap().3 += n;
        }
        let choice = self.choice_from_uct(root, false);
        let next = self.h[&root].1[choice];
        move_diff(root, next)
    }

    pub fn search_one(&mut self, code: i32) -> (f64, f64) {
        if self.h[&code].1.len() == 0 {
            self.expand(code)
        } else {
            let choice = self.choice_from_uct(code, true);
            let (dw, dn) = self.search_one(self.h[&code].1[choice]);
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
                                       uct(turn, w, np, ni, c_flg)
                                   }).collect::<Vec<f64>>();
        let mut choice = 0;
        let mut u_max = std::f64::MIN;
        if !c_flg {
            println!("{:?}", v_uct);
        }
        for (i, &u) in v_uct.iter().enumerate() {
            if u > u_max {
                choice = i;
                u_max = u;
            }
        }
        choice
    }
}