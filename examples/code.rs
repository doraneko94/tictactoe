use tictactoe::game::*;

fn main() {
    let mut env = Game::new();
    let mut code = env.code;
    let (c, _) = env.next(code, 0).unwrap();
    code = c;
    
    let (c, _) = env.next(code, 7).unwrap();
    println!("{}", move_diff(code, c));
    
}