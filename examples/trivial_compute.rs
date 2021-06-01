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

    my_shader! {compute = {
        [group1 [buffer loop in] uint[]] indices;
        [group2 [buffer in out] uint[]] indices2;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                indices2[index] = indices[index]+indices2[index];
            }
        }}
    }}

    const S: ComputeShader = eager_compute_shader! {compute!()};

    eager_binding! {context = compute!()};

    let program = compile(&S, &device, context.get_layout(&device)).await;

    let indices_1_data = BufferData::new(vec![1, 2, 3, 4]);
    let indices_1 = BindGroup1::new(&device, &indices_1_data);

    let indices_2_data = BufferData::new(vec![2, 2, 2, 2]);
    let indices_2 = BindGroup1::new(&device, &indices_2_data);

    let indices_3_data = BufferData::new(vec![4, 3, 2, 1]);
    let indices_3 = BindGroup1::new(&device, &indices_3_data);

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&program.pipeline);
            let context1 = context.set_indices(&mut cpass, &indices_3);
            {
                let context2 = (&context1).set_indices2(&mut cpass, &indices_1);

                context2.runnable(|| compute_run(&mut cpass, 4));
            }
            {
                let context3 =
                    context1.set_indices2(&mut cpass, &indices_2);
                {
                    context3.runnable(|| compute_run(&mut cpass, 4));
                }
            }
        }

        let x = indices_1.setup_read_0(&device, &mut encoder, 0..16);
        let y = indices_2.setup_read_0(&device, &mut encoder, 0..16);

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
