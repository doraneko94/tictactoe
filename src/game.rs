use ndarray::*;
use crate::mchs::*;
use crate::mchs_net::*;
use crate::net::*;

pub const LINES: [(usize, usize, usize); 8] = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)];

pub fn zip_code(board: &Array2<i8>) -> i32 {
    let mut code = 0;
    for i in 0..9 {
        let j = 8 - i;
        let y = j / 3;
        let x = j % 3;
        code <<= 2;
        code += board[[y, x]] as i32 + 1;
    }
    code
}
 
pub fn unzip_code(code: i32) -> Array2<i8> {
    let mut b = code;
    let mut board = Array2::zeros((3, 3));
    for i in 0..9 {
        let y = i / 3;
        let x = i % 3;
        let c = b % 4;
        b >>= 2;
        board[[y, x]] = c as i8 - 1;
    }
    board
}

pub fn flip_code(code: i32) -> i32 {
    let mut ret = code;
    for i in 0..3 {
        let j = i * 3 * 2;
        let tmp = code >> j;
        let l = tmp % 4;
        let r = (tmp >> 4) % 4;
        ret -= l << j;
        ret += r << j;
        ret -= r << (j + 4);
        ret += l << (j + 4);
    }
    ret
}

pub fn turn(code: i32) -> i8 {
    ((code >> 9 * 2) % 4 - 1) as i8
}

pub fn flip_turn(mut code: i32) -> i32 {
    let now = turn(code);
    code -= 2 * (now as i32) << 9 * 2;
    code
}

pub fn move_code(mut code: i32, mov: usize) -> i32 {
    let turn = turn(code);
    code += (turn as i32) << (mov * 2);
    code
}

pub fn move_diff(mut prev: i32, mut post: i32) -> usize {
    for i in 0..9 {
        if prev % 4 != post % 4 {
            return i;
        }
        prev >>= 2;
        post >>= 2;
    }
    panic!();
}

pub fn next_code(code: i32, mov: usize) -> i32 {
    let c = move_code(code, mov);
    flip_turn(c)
}

pub fn xo(token: i8) -> &'static str {
    match token {
        -1 => "O",
        1 => "X",
        _ => " "
    }
}

pub fn valid(code: i32, mov: usize) -> bool {
    if mov >= 9 { return false; }
    let pos = (code >> (mov * 2)) % 4;
    match pos {
        1 => true,
        _ => false,
    }
}

pub fn winner(code: i32) -> i8 {
    let c = (0..9).map(|i| (code >> i * 2) % 4).collect::<Vec<i32>>();
    for l in LINES.iter() {
        let x0 = c[l.0];
        let x1 = c[l.1];
        let x2 = c[l.2];
        let sum = x0 + x1 + x2;
        if sum == 6 { return 1; }
        else if sum == 0 { return -1 };
    }
    for &i in c.iter() {
        if i == 1 {
            return 2;
        }
    }
    0
}

pub struct Game {
    pub code: i32,
}

impl Game {
    pub fn new() -> Self {
        let mut code = 0;
        for _ in 0..9 {
            code <<= 2;
            code += 1;
        }
        code += 2 << 9 * 2;
        Self { code }
    }

    pub fn init(&mut self) {
        self.code = 0;
        for _ in 0..9 {
            self.code <<= 2;
            self.code += 1;
        }
        self.code += 2 << 9 * 2;
    }

    pub fn view(&self) {
        let board = unzip_code(self.code);
        println!("-------------");
        println!("| {} | {} | {} |", xo(board[[0, 0]]), xo(board[[0, 1]]), xo(board[[0, 2]]));
        println!("-------------");
        println!("| {} | {} | {} |", xo(board[[1, 0]]), xo(board[[1, 1]]), xo(board[[1, 2]]));
        println!("-------------");
        println!("| {} | {} | {} |", xo(board[[2, 0]]), xo(board[[2, 1]]), xo(board[[2, 2]]));
        println!("-------------");
    }

    pub fn next(&mut self, mut code: i32, mov: usize) -> Result<(i32, i8), &str> {
        let flip = flip_code(code);
        let flipped = if self.code == code {
            false
        } else if self.code == flip {
            true
        } else {
            panic!();
        };
        
        if !valid(code, mov) { Err("Invalid move!") }
        else {
            code = next_code(code, mov);
            if flipped { self.code = flip_code(code); }
            else { self.code = code; }
            Ok((code, winner(self.code)))
        }
    }

    pub fn player(&mut self) -> Result<(i32, i8), &str>  {
        self.view();
        let turn = turn(self.code);
        println!("Turn '{}', move: ", xo(turn as i8));
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).ok();
        let mov: usize = s.trim().parse().ok().unwrap();
        let code = self.code;
        self.next(code, mov)
    }

    pub fn cpu(&mut self, mchs: &mut MCHS) -> Result<(i32, i8), &str>  {
        self.view();
        let turn = turn(self.code);
        println!("Turn '{}', move: ", xo(turn as i8));
        
        let code = self.code;
        let mov: usize = mchs.search(code, 16000);
        println!("{}", mov);
        self.next(code, mov)
    }

    pub fn cpu_net(&mut self, mchs: &mut MCHSNet, net: &Net, train: bool) -> Result<(i32, i8), &str>  {
        let code = self.code;
        let mov: usize = mchs.search(code, 160, net, train);
        self.next(code, mov)
    }

    pub fn cpu_net_vs(&mut self, mchs: &mut MCHSNet, net: &Net, train: bool) -> Result<(i32, i8), &str>  {
        self.view();
        let turn = turn(self.code);
        println!("Turn '{}', move: ", xo(turn as i8));
        
        let code = self.code;
        let mov: usize = mchs.search(code, 160, net, train);
        println!("{}", mov);
        self.next(code, mov)
    }
}