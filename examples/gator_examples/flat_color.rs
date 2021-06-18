#![recursion_limit = "1024"]
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
    GraphicsShader,
};

use crate::pipeline::AbstractBind;
pub use pipeline::bind::{BindGroup1, BindGroup2, BufferData, Indices, Vertex};

pub use pipeline::helper::{
    generate_identity_matrix, generate_projection_matrix, generate_view_matrix, load_model,
    rotation_x, rotation_y, rotation_z, scale, translate,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

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

    my_shader! { vertex = {
        [[vertex in] vec3] a_position;
        [group1 [uniform in] mat4] u_view;
        [group1 [uniform in] mat4] u_proj;
        [group2 [uniform in] mat4] u_model;

        [[out] vec4] gl_Position;
        {{
            void main() {
                gl_Position = u_proj * u_view * u_model * vec4( a_position, 1.0);
            }
        }}
    }}

    my_shader! { fragment = {
        [[out] vec4] color;
        {{
            void main() {
                color = vec4(1.0, 0.4, 0.25, 1.0);
            }
        }}
    }}

    const S_V: GraphicsShader = eager_graphics_shader! {vertex!()};
    const S_F: GraphicsShader = eager_graphics_shader! {fragment!()};

    eager_binding! {context = vertex!(), fragment!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, S_V, S_F, GraphicsCompileArgs::default());

    let (position_data, _, index_data) = load_model("src/models/teapot.obj");

    let positions = Vertex::new(&device, &BufferData::new(position_data));
    let indices = Indices::new(&device, &index_data);

    let view_mat = BufferData::new(generate_view_matrix());

    let proj_mat = BufferData::new(generate_projection_matrix(
        size.width as f32 / size.height as f32,
    ));

    let mut model_mat = generate_identity_matrix();

    let bind_group_view_proj = BindGroup2::new(&device, &view_mat, &proj_mat);

    // A "chain" of buffers that we render on to the display
    let swap_chain = generate_swap_chain(&surface, &window, &device);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_current_frame()
                    .expect("Timeout when acquiring next swap chain texture")
                    .output;

                let mut init_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                model_mat = rotation_y(model_mat, 0.05);
                let bind_group_model = BindGroup1::new(&device, &BufferData::new(model_mat));

                {
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

                        let context1 = (&context).set_a_position(&mut rpass, &positions);

                        {
                            let context2 =
                                (&context1).set_u_view_u_proj(&mut rpass, &bind_group_view_proj);

                            {
                                let context3 =
                                    (&context2).set_u_model(&mut rpass, &bind_group_model);
                                {
                                    context3
                                        .runnable(|| graphics_run_indices(&mut rpass, &indices, 1));
                                }
                            }
                        }
                    }
                    queue.submit(Some(init_encoder.finish()));
                }
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
    /* let window = winit::window::Window::new(&event_loop).unwrap(); */

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
