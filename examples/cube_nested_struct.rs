#![recursion_limit = "1024"]
#![feature(const_generics)]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header::{
    generate_swap_chain, graphics_run_indices, setup_render_pass, GraphicsCompileArgs,
};

pub use pipeline::bind::{BindGroup1, BufferData, Indices, Vertex};
pub use pipeline::AbstractBind;

pub use pipeline::helper::{generate_projection_matrix, generate_view_matrix, load_cube};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    // Create a surface to draw images on
    // this is the new way wgpu does things... unsafe is kind of sad
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
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

    my_shader! {pipeline = {
        struct VertexOutput {
            [[location(0)]] fragmentColor: vec3<f32>;
            [[builtin(position)]] position: vec4<f32>;
        };

        struct InnerLocals {
            transform: mat4x4<f32>;
        };

        [[block]]
        struct Locals {
            inner: InnerLocals;
        };
        [[group(0), binding(0)]]
        var r_locals: Locals;

        [[stage(vertex)]]
        fn vs_main(
            [[location(0)]] position: vec3<f32>,
            [[location(1)]] color: vec3<f32>,
        ) -> VertexOutput {
            var out: VertexOutput;
            out.fragmentColor = color;
            out.position = r_locals.inner.transform * vec4<f32>(position.x, position.y, position.z, 1.0);
            return out;
        }

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
            return vec4<f32>(in.fragmentColor.x,in.fragmentColor.y, in.fragmentColor.z, 1.0);
        }
    }}

    eager_binding! {vertex = "vs_main", fragment = "fs_main"; let context = pipeline!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, GraphicsCompileArgs::default());

    let (positions, _, index_data) = load_cube();

    let color_data = BufferData::new(vec![
        [0.583, 0.771, 0.014],
        [0.609, 0.115, 0.436],
        [0.327, 0.483, 0.844],
        [0.822, 0.569, 0.201],
        [0.435, 0.602, 0.223],
        [0.310, 0.747, 0.185],
        [0.597, 0.770, 0.761],
        [0.559, 0.436, 0.730],
        [0.359, 0.583, 0.152],
        [0.483, 0.596, 0.789],
        [0.559, 0.861, 0.639],
        [0.195, 0.548, 0.859],
        [0.014, 0.184, 0.576],
        [0.771, 0.328, 0.970],
        [0.406, 0.615, 0.116],
        [0.676, 0.977, 0.133],
        [0.971, 0.572, 0.833],
        [0.140, 0.616, 0.489],
        [0.997, 0.513, 0.064],
        [0.945, 0.719, 0.592],
        [0.543, 0.021, 0.978],
        [0.279, 0.317, 0.505],
        [0.167, 0.620, 0.077],
        [0.347, 0.857, 0.137],
        [0.055, 0.953, 0.042],
        [0.714, 0.505, 0.345],
        [0.783, 0.290, 0.734],
        [0.722, 0.645, 0.174],
        [0.302, 0.455, 0.848],
        [0.225, 0.587, 0.040],
        [0.517, 0.713, 0.338],
        [0.053, 0.959, 0.120],
        [0.393, 0.621, 0.362],
        [0.673, 0.211, 0.457],
        [0.820, 0.883, 0.371],
        [0.982, 0.099, 0.879],
    ]);

    let locals = BindGroup1::new(
        &device,
        &Locals {
            inner: InnerLocals {
                transform: generate_projection_matrix(size.width as f32 / size.height as f32)
                    * generate_view_matrix(),
            },
        },
    );

    let vertex_position = Vertex::new(&device, &BufferData::new(positions));
    let vertex_color = Vertex::new(&device, &color_data);
    let indices = Indices::new(&device, &index_data);

    // A "chain" of buffers that we render on to the display
    let swap_chain = generate_swap_chain(&surface, &window, &device);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut init_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let frame = swap_chain
                    .get_current_frame()
                    .expect("Timeout when acquiring next swap chain texture")
                    .output;
                {
                    let mut rpass = setup_render_pass(
                        &program,
                        &mut init_encoder,
                        wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: true,
                                },
                            }],
                            depth_stencil_attachment: None,
                        },
                    );

                    //
                    // Handle the setup for the pipeline
                    //

                    let context1 = (&context).set_position(&mut rpass, &vertex_position);

                    {
                        let context2 = (&context1).set_color(&mut rpass, &vertex_color);
                        {
                            let context3 = context2.set_r_locals(&mut rpass, &locals);
                            {
                                let _ = context3
                                    .runnable(&mut rpass, |r| graphics_run_indices(r, &indices, 1));
                            }
                        }
                    }
                }
                queue.submit(Some(init_encoder.finish()));
            }
            // When the window closes we are done. Change the status
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            // Ignore any other types of events
            _ => {}
        }
    });
}

fn main() {
    // From examples of wgpu-rs, set up a window we can use to view our stuff
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
