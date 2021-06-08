use zerocopy::AsBytes as _;
pub trait Alignment {
    // For some data types, the total size they take up is larger than the amount of data used because backends will add padding for alignment.
    fn alignment_size() -> usize;
    fn align_bytes(&self) -> &[u8];
}

impl Alignment for f32 {
    fn alignment_size() -> usize {
        std::mem::size_of::<f32>()
    }
    fn align_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}


impl Alignment for cgmath::Matrix4<f32>{
    fn alignment_size() -> usize {
        64
    }
    fn align_bytes(&self) -> &[u8] {
        let mat_slice: &[f32; 16] = self.as_ref();
        bytemuck::cast_slice(mat_slice.as_bytes())
    }
}