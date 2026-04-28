/// A grayscale image buffer.
#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    data: Vec<u8>,
}

impl Image {
    /// Creates a new image buffer with a manually specified stride.
    pub fn new(width: usize, height: usize, stride: usize) -> Self {
        let size = height * stride;

        Self {
            width,
            height,
            stride,
            data: vec![0; size],
        }
    }

    /// Creates a new image, automatically padding the stride to the next
    /// multiple of 16. This ensures every row starts at a 16-byte aligned
    /// boundary, which allows LLVM to heavily optimize loops across rows
    /// using ARM NEON SIMD instructions.
    pub fn new_simd_aligned(width: usize, height: usize) -> Self {
        let stride = (width + 15) & !15;
        Self::new(width, height, stride)
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[inline(always)]
    pub fn row(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }
}
