use tictactoe::game::*;

fn main() {
    let mut env = Game::new();
    let winner;
    println!("{}", env.code);
    loop {
        match env.player() {
            Ok((_, w)) => {
                if w != 2 {
                    winner = w;
                    break;
                }
            }
            Err(s) => { println!("{}", s); }
        }
    };
    env.view();
    if winner == 0 { println!("Draw.."); }
    else { println!("{} wins!", xo(winner)); }
}