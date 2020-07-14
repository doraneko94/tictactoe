use tictactoe::game::*;
use tictactoe::mchs::*;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

fn main() {
    let mut env = Game::new();
    let player_turn = match Uniform::new(0usize, 2usize).sample(&mut thread_rng()) {
        0 => 1,
        1 => -1,
        _ => panic!(),
    };
    let mut mchs = MCHS::new(player_turn * -1);
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
            match env.cpu(&mut mchs) {
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