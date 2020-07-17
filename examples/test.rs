use tictactoe::game::*;
use anyhow::Result;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};
use tictactoe::net::Net;
use tictactoe::mchs_net::{MCHSNet, SaveData};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

fn main() {
    let mut env = Game::new();
    let player_turn = match Uniform::new(0usize, 2usize).sample(&mut thread_rng()) {
        0 => 1,
        1 => -1,
        _ => panic!(),
    };
    let mut mchs = MCHSNet::new(player_turn * -1);
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let _ = vs.load("save.nn");
    let net = Net::new(&vs);
    println!("player's mark is {}", xo(player_turn));
    let winner;
    loop {
        let turn = turn(env.code);
        if player_turn == turn {
            match env.player() {
                Ok((c, w)) => {
                    if w != 2 {
                        winner = w;
                        break;
                    }
                    mchs.new_node(c);
                }
                Err(s) => { println!("{}", s); }
            }
        } else {
            match env.cpu_net_vs(&mut mchs, &net, false) {
                Ok((_, w)) => {
                    if w != 2 {
                        winner = w;
                        break;
                    }
                }
                Err(_) => { panic!(); }
            }
        }
    };
    env.view();
    if winner == 0 { println!("Draw.."); }
    else { println!("{} wins!", xo(winner)); }
}