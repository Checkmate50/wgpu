use zerocopy::AsBytes as _;
pub trait Alignment {
    // For some data types, the total size they take up is larger than the amount of data used because backends will add padding for alignment.
    fn alignment_size() -> usize;
    fn align_bytes(&self) -> Vec<u8>;
}

impl Alignment for f32 {
    fn alignment_size() -> usize {
        std::mem::size_of::<f32>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

impl Alignment for Vec<u32> {
    fn alignment_size() -> usize {
        std::mem::size_of::<[u32; 1]>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        self.as_slice().as_bytes().to_vec()
    }
}

impl Alignment for Vec<f32> {
    fn alignment_size() -> usize {
        std::mem::size_of::<[f32; 1]>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        self.as_slice().as_bytes().to_vec()
    }
}

impl Alignment for Vec<[f32; 2]> {
    fn alignment_size() -> usize {
        std::mem::size_of::<[f32; 2]>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        numbers.as_slice().as_bytes().to_vec()
    }
}

// [f32; 3] gets converted to size [f32; 4] for alignment purposes
impl Alignment for Vec<[f32; 3]> {
    fn alignment_size() -> usize {
        std::mem::size_of::<[f32; 4]>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| {
                let mut y = x.to_vec();
                y.push(0.0);
                y
            }) // We need to extend Vec3 -> Vec4 for alignment
            .flatten()
            .collect();
        numbers.as_slice().as_bytes().to_vec()
    }
}

impl Alignment for Vec<[f32; 4]> {
    fn alignment_size() -> usize {
        std::mem::size_of::<[f32; 4]>()
    }
    fn align_bytes(&self) -> Vec<u8> {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        numbers.as_slice().as_bytes().to_vec()
    }
}

impl Alignment for cgmath::Matrix4<f32> {
    fn alignment_size() -> usize {
        64
    }
    fn align_bytes(&self) -> Vec<u8> {
        let mat_slice: &[f32; 16] = self.as_ref();
        bytemuck::cast_slice(mat_slice.as_bytes()).to_vec()
    }
}
