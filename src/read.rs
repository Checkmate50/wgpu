pub struct MyBufferView {
    read_buffer: wgpu::Buffer,
    bounds: std::ops::Range<u64>,
}

impl Drop for MyBufferView {
    fn drop(&mut self) {
        self.read_buffer.unmap()
    }
}

impl<'a> MyBufferView {
    pub fn new(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, bounds: std::ops::Range<u64>) -> Self {
        assert!(!bounds.is_empty());

        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_reading_buffer"),
            size: bounds.end - bounds.start,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            src,
            bounds.start,
            &read_buffer,
            0,
            bounds.end - bounds.start,
        );

        MyBufferView {
            read_buffer,
            bounds
        }
    }

    pub async fn read(&'a self, device: &wgpu::Device) -> Result<wgpu::BufferView<'a>, &'static str>{
        let buffer_slice = self
            .read_buffer
            .slice(0..(self.bounds.end - self.bounds.start));
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let x = buffer_slice.get_mapped_range();
            return Ok(x);
        } else {
            return Err("failed to read compute buffer on gpu!");
        }
    }
}
