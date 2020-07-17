use anyhow::Result;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};
use tictactoe::net::Net;
use tictactoe::mchs_net::{MCHSNet, SaveData};
use tictactoe::game::*;

fn main() {
    loop {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let mut net = Net::new(&vs);

        let mut mchs0 = MCHSNet::new(1);
        let mut mchs1 = MCHSNet::new(-1);

        let mut env = Game::new();
        let mut winner = 0;

        let mut save = SaveData::new();

        for i in 1..101 {
            if i % 10 == 0 {
                println!("game = {}", i);
            }
            loop {
                let turn = turn(env.code);
                if turn == 1 {
                    match env.cpu_net(&mut mchs0, &net, true) {
                        Ok((c, w)) => {
                            if w != 2 {
                                winner = w;
                                break;
                            }
                            mchs1.receive(c);
                        }
                        Err(s) => { panic!(); }
                    }
                } else {
                    match env.cpu_net(&mut mchs1, &net, true) {
                        Ok((c, w)) => {
                            if w != 2 {
                                winner = w;
                                break;
                            }
                            mchs0.receive(c);
                        }
                        Err(_) => { panic!(); }
                    }
                }
            };
            save.save(&mchs0, winner);
            mchs0.init(1);
            mchs1.init(-1);
            env.init();
        }
        let dataset = save.to_dataset();
        net.train(&dataset);
        let mut vs_old = nn::VarStore::new(Device::cuda_if_available());
        match vs_old.load("save.nn") {
            Ok(_) => {
                let mut net_old = Net::new(&vs_old);

                let mut env = Game::new();
                let mut winner = 0;

                let mut n_games = 0;
                let mut n_wins = 0;

                for i in 0..20 {
                    let turn0 = if i % 2 == 0 {
                        1
                    } else {
                        -1
                    };
                    let mut mchs0 = MCHSNet::new(turn0);
                    let mut mchs1 = MCHSNet::new(turn0 * -1);

                    loop {
                        let turn = turn(env.code);
                        if turn == turn0 {
                            match env.cpu_net(&mut mchs0, &net, false) {
                                Ok((c, w)) => {
                                    if w != 2 {
                                        winner = w;
                                        break;
                                    }
                                    mchs1.receive(c);
                                }
                                Err(s) => { panic!(); }
                            }
                        } else {
                            match env.cpu_net(&mut mchs1, &net_old, false) {
                                Ok((c, w)) => {
                                    if w != 2 {
                                        winner = w;
                                        break;
                                    }
                                mchs0.receive(c);
                                }
                                Err(_) => { panic!(); }
                            }
                        }
                    };
                    if winner == turn0 {
                        n_games += 1;
                        n_wins += 1;
                        println!("win");
                    } else if winner == turn0 * -1 {
                        n_games += 1;
                        println!("lose");
                    } else {
                        println!("draw");
                    }
                    env.init();
                }
                if n_games > 0 && n_wins as f32 / n_games as f32 >= 0.6 {
                    let _ = vs.save("save.nn");
                    println!("saved!");
                } else {
                    println!("not saved..");
                }
            }
            Err(_) => {
                let _ = vs.save("save.nn");
                println!("saved!");
            }
        }
    }
}