#![recursion_limit = "256"]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

pub use pipeline::wgpu_compute_header::{compile, compute_run, ComputeShader};

pub use pipeline::bind::{BindGroup1, BufferData, Indices, Vertex};
pub use pipeline::AbstractBind;

use std::convert::TryInto;
use std::rc::Rc;
use zerocopy::AsBytes;

async fn execute_gpu() {
    // qualifiers
    // buffer: is a buffer?
    // in: this parameter must be bound to before the program runs
    //     thus it can be read inside of the program scope
    // out: if this parameter is bound, it must be rebound after each iteration
    //      Only out variables can be mutated
    //      Only out variables can be read as a result of the program
    //      If out has been unassigned then an error is raised when it is read
    // loop: one or more of these loop annotations are required per program. Atm, the values bound is assumed to be of equal length and this gives the number of iterations(gl_GlobalInvocationID.x)
    //      the size of any out buffers that need to be created

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropiate adapter");

    // The device manages the connection and resources of the adapter
    // The queue is a literal queue of tasks for the gpu
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let queue = Rc::new(queue);

    my_shader! {trivial = {
        [group1 [buffer loop in out] uint[]] indices;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            uint collatz_iterations(uint n) {
                uint i = 0;
                while(n > 1) {
                    if (mod(n, 2) == 0) {
                        n = n / 2;
                    }
                    else {
                        n = (3 * n) + 1;
                    }
                    i++;
                }
                return i;
            }

            void main() {
                uint index = gl_GlobalInvocationID.x;
                indices[index] = collatz_iterations(indices[index]);
            }
        }}
    }}

    const S: ComputeShader = eager_compute_shader! {trivial!()};
    eager_binding! {context = trivial!()};

    let program = compile(&S, &device, context.get_layout(&device)).await;

    let indices = BufferData::new(vec![0, 0, 0, 0]);

    let bg_i = BindGroup1::new(&device, &indices);

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let y = bg_i.setup_write_0(&device, 0..16);
        y.write(&device).await.unwrap().copy_from_slice(vec![1,2,3,4].as_bytes());
        y.collect(&mut encoder);

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&program.pipeline);
            let context1 = context.set_indices(&mut cpass, &bg_i);
            {
                context1.runnable(|| compute_run(&mut cpass, 4));
            }
        }

        let x = bg_i.setup_read_0(&device, &mut encoder, 0..16);

        queue.submit(Some(encoder.finish()));

        println!(
            "{:?}",
            x.read(&device)
                .await
                .unwrap()
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<u32>>()
        );
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
