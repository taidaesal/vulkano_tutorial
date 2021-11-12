// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

use std::fmt;

pub struct RawFace {
    pub verts: [usize; 3],
    pub norms: Option<[usize; 3]>,
    pub text: Option<[usize; 3]>,
}

impl RawFace {
    // call with invert = true if the models are using a clockwise winding order
    //
    // Blender files are a common example
    pub fn new(raw_arg: &str, invert: bool) -> RawFace {
        let arguments: Vec<&str> = raw_arg.split_whitespace().collect();
        RawFace {
            verts: RawFace::parse(arguments.clone(), 0, invert).unwrap(),
            norms: RawFace::parse(arguments.clone(), 2, invert),
            text: RawFace::parse(arguments.clone(), 1, invert),
        }
    }

    fn parse(inpt: Vec<&str>, index: usize, invert: bool) -> Option<[usize; 3]> {
        let f1: Vec<&str> = inpt.get(0).unwrap().split("/").collect();
        let f2: Vec<&str> = inpt.get(1).unwrap().split("/").collect();
        let f3: Vec<&str> = inpt.get(2).unwrap().split("/").collect();
        let a1 = f1.get(index).unwrap().clone();
        let a2 = f2.get(index).unwrap().clone();
        let a3 = f3.get(index).unwrap().clone();
        match a1 {
            "" => None,
            _ => {
                let p1: usize = a1.parse().unwrap();
                let (p2, p3): (usize, usize) = if invert {
                    (a3.parse().unwrap(), a2.parse().unwrap())
                } else {
                    (a2.parse().unwrap(), a3.parse().unwrap())
                };
                Some([p1 - 1, p2 - 1, p3 - 1]) // .obj files aren't 0-index
            }
        }
    }
}

impl fmt::Display for RawFace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verts = format!("{}/{}/{}", self.verts[0], self.verts[1], self.verts[2]);
        let normals = match self.norms {
            None => "None".to_string(),
            Some(x) => {
                format!("{}/{}/{}", x[0], x[1], x[2])
            }
        };
        let textures = match self.text {
            None => "None".to_string(),
            Some(x) => {
                format!("{}/{}/{}", x[0], x[1], x[2])
            }
        };
        write!(
            f,
            "Face:\n\tVertices: {}\n\tNormals: {}\n\tTextures: {}",
            verts, normals, textures
        )
    }
}
