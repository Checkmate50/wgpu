mod wgpu_compute_header;

use std::{convert::TryInto, str::FromStr};

use zerocopy::AsBytes as _;

async fn run() {
    let numbers = if std::env::args().len() == 1 {
        let default = vec![1, 2, 3, 4];
        log::info!("No numbers were provided, defaulting to {:?}", default);
        default
    } else {
        std::env::args()
            .skip(1)
            .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
            .collect()
    };

    print!("Starting numbers: {:?}\n", numbers);
    // To see the output, run `RUST_LOG=info cargo run --example hello-compute`.
    print!("Times: {:?}", execute_gpu(numbers).await);
}

async fn execute_gpu(numbers: Vec<u32>) -> Vec<u32> {
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    /*     let s = shader! {
        [buffer uint[]] indices;
        [buffer uint[]] indices2;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    }; */
    let s = shader! {
        [buffer uint[]] indices;
        [buffer uint[]] indices2;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    };

    let (program, mut encoder, bindings) = wgpu_compute_header::compile(s).await;

    let staging_buffer = program.device.create_buffer_with_data(
        numbers.as_slice().as_bytes(),
        wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    );

    let storage_buffer = program.device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
        label: None,
    });

    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
    {
        let new_bindings =
            wgpu_compute_header::bind(&bindings, &storage_buffer, size, "indices".to_string());

        let final_bindings =
            wgpu_compute_header::bind(&new_bindings, &storage_buffer, size, "indices2".to_string());
        {
            wgpu_compute_header::run(&mut encoder, &program, final_bindings);
        }
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    program.queue.submit(&[encoder.finish()]);

    // Note that we're not calling `.await` here.
    let buffer_future = staging_buffer.map_read(0, size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    program.device.poll(wgpu::Maintain::Wait);

    if let Ok(mapping) = buffer_future.await {
        mapping
            .as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    env_logger::init();

    futures::executor::block_on(run());
}
