use std::rc::Rc;

pub struct MyBufferViewMut {
    src: Rc<wgpu::Buffer>,
    write_buffer: wgpu::Buffer,
    bounds: std::ops::Range<u64>,
}

impl MyBufferViewMut {
    pub fn new(
        device: &wgpu::Device,
        src: Rc<wgpu::Buffer>,
        bounds: std::ops::Range<u64>,
    ) -> Self {
        let write_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_writing_buffer"),
            size: bounds.end - bounds.start,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });

        MyBufferViewMut {
            write_buffer,
            bounds,
            src,
        }
    }

    pub async fn write(
        &self,
        device: &wgpu::Device,
    ) -> Result<wgpu::BufferViewMut<'_>, &'static str> {
        let buffer_slice = self.write_buffer.slice(0..(self.bounds.end - self.bounds.start));

        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);
        device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let x = buffer_slice.get_mapped_range_mut();
            return Ok(x);
        } else {
            return Err("failed to read compute buffer on gpu!");
        }
    }

    pub fn collect(self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.write_buffer,
            0,
            &self.src,
            self.bounds.start,
            self.bounds.end - self.bounds.start,
        );
    }
}

impl Drop for MyBufferViewMut {
    fn drop(&mut self) {
        /* panic!("This struct acts lazily so if this struct is dropped without being collected then none of your changes made it through.") */
    }
}
