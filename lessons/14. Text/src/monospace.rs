// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

use png;

use std::io::Cursor;

pub struct Monospace {
    input_width: usize,
    texture_data: Vec<u8>,
}

impl Monospace {
    pub fn new() -> Monospace {
        let png_bytes = include_bytes!("./textures/texture_small.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let width = info.width;
        let height = info.height;
        let mut image_data = Vec::new();
        let depth: u32 = match info.bit_depth {
            png::BitDepth::One => 1,
            png::BitDepth::Two => 2,
            png::BitDepth::Four => 4,
            png::BitDepth::Eight => 8,
            png::BitDepth::Sixteen => 16,
        };
        image_data.resize((width * height * depth) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        Monospace {
            input_width: width as usize,
            texture_data: image_data,
        }
    }

    pub fn text(&self, value: &str) -> (Vec<u8>, u32, u32) {
        let chars = value.chars();
        let length = value.len();
        let height = 64;
        let width = length * 64;
        let mut ret: Vec<u8> = Vec::new();
        ret.resize(height * width * 4, 0); // each pixel has four components
        for (i, ch) in chars.enumerate() {
            let val = ch.to_digit(10).unwrap() as usize;
            let source_index = val * 64; // The pixel index of the left pixel column
            let target_index = i * 64;
            for row in 0..64 {
                let source_row_offset = row * self.input_width * 4;
                let target_row_offset = row * width * 4;
                for col in 0..64 {
                    let source_col_offset = (source_index + col) * 4;
                    let target_col_offset = (target_index + col) * 4;
                    for ii in 0..4 {
                        ret[target_row_offset + target_col_offset + ii] =
                            self.texture_data[source_row_offset + source_col_offset + ii];
                    }
                }
            }
        }
        (ret, 64, (length * 64) as u32)
    }
}
