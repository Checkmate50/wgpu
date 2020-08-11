#[macro_use]
extern crate pipeline;

use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

fn main() {
    let input = BufReader::new(File::open("examples/models/sphere.obj").unwrap());
    let mut dome: Obj = load_obj(input).unwrap();
    let mut vertices_vec = Vec::new();
    for i in &dome.vertices {
        vertices_vec.push(i.position);
    }

    println!("{:?}\n\n", dome.vertices);

    println!("{:?}", dome.indices);
}
