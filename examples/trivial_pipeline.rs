#![recursion_limit = "256"]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

pub use pipeline::wgpu_compute_header::{compile, compute_run, ComputeShader};

pub use pipeline::bind::{BindGroup1, BufferData, Indices, Vertex};
pub use pipeline::AbstractBind;

use std::convert::TryInto;

async fn execute_gpu() {
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

    //todo can parameters that are only out be inferred?
    my_shader! {One = {
        [group1 [buffer loop in] uint[]] add_one_in;
        [group2 [buffer in out] uint[]] add_two_in;
        //unit ->
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_in[index] = add_one_in[index]+1;
            }
        }}
    }}

    my_shader! {Two = {
        [group1 [buffer loop in] uint[]] add_two_in;
        [group2 [buffer in out] uint[]] add_two_result;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_result[index] = add_two_in[index]+2;
            }
        }}
    }}

    const ADD_ONE: ComputeShader = eager_compute_shader! {One!()};

    eager_binding! {context = One!()};

    const ADD_TWO: ComputeShader = eager_compute_shader! {Two!()};

    eager_binding! {next_context = Two!()};

    let program1 = compile(&ADD_ONE, &device, context.get_layout(&device)).await;

    let program2 = compile(&ADD_TWO, &device, context.get_layout(&device)).await;

    let indices = BindGroup1::new(&device, &BufferData::new(vec![1, 2, 3, 4]));
    let empty1 = BindGroup1::new(&device, &BufferData::new(vec![0, 0, 0, 0]));
    let empty2 = BindGroup1::new(&device, &BufferData::new(vec![0, 0, 0, 0]));

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&program1.pipeline);
            {
                let context1 = context.set_add_one_in(&mut cpass, &indices);
                {
                    let context2 = context1.set_add_two_in(&mut cpass, &empty1);
                    {
                        context2.runnable(|| compute_run(&mut cpass, 4));
                    }
                }
            }
            cpass.set_pipeline(&program2.pipeline);
            {
                let next_context1 = next_context.set_add_two_in(&mut cpass, &empty1);

                {
                    let next_context2 = next_context1.set_add_two_result(&mut cpass, &empty2);

                    {
                        next_context2.runnable(|| compute_run(&mut cpass, 4));
                    }
                }
            }
        }
        let x = empty1.setup_read_0(&device, &mut encoder, 0..16);
        let y = empty2.setup_read_0(&device, &mut encoder, 0..16);

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
        println!(
            "{:?}",
            y.read(&device)
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
