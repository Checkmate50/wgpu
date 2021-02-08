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
    generate_swap_chain, graphics_run_indices, setup_render_pass, GraphicsShader, PipelineType,
};

use crate::pipeline::AbstractBind;
pub use pipeline::bind::{BindGroup1, BindGroup2, Indices, Vertex};

pub use pipeline::helper::{
    generate_identity_matrix, generate_projection_matrix, generate_view_matrix, load_model,
    rotate_vec3, rotation_y, scale,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
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
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    my_shader! {vertex = {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;
        [group1 [uniform in] vec3] Ambient;
        [group2 [uniform in] vec3] LightDirection;
        [group3 [uniform in] mat4] u_view;
        [group3 [uniform in] mat4] u_proj;
        [group4 [uniform in] mat4] u_model;
        [[] int] gl_VertexID;

        [[out] vec3] fragmentNormal;
        [[out] vec4] gl_Position;
        {{
            void main() {

                vec4 worldNormal = vec4(a_normal, 0.0) * inverse(u_model) * inverse(u_view);

                fragmentNormal = worldNormal.xyz;

                gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
            }
        }}
    }}

    my_shader! {fragment = {
        [[in] vec3] fragmentNormal;
        [group1 [uniform in] vec3] Ambient;
        [group2 [uniform in] vec3] LightDirection;
        [[out] vec4] color;
        {{
            void main() {
                vec3 fragColor = vec3(1.0, 0.0, 0.0);
                color = vec4(Ambient + fragColor * max(dot(normalize(fragmentNormal), normalize(LightDirection)), 0.0), 1.0);
            }
        }}
    }}

    const S_V: GraphicsShader = eager_graphics_shader! {vertex!()};

    const S_F: GraphicsShader = eager_graphics_shader! {fragment!()};

    eager_binding! {context = vertex!(), fragment!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, S_V, S_F, PipelineType::Color);

    let (positions, normals, index_data) = load_model("src/models/teapot.obj");

    let mut light_direction = vec![[20.0, 0.0, 0.0]];

    let light_ambient = vec![[0.1, 0.0, 0.0]];

    let view_mat = generate_view_matrix();

    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);

    let model_mat_init = generate_identity_matrix();
    let mut model_mat = scale(model_mat_init, 0.7);

    let vertex_position = Vertex::new(&device, &positions);
    let vertex_normal = Vertex::new(&device, &normals);
    let indices = Indices::new(&device, &index_data);
    let bind_group_ambient = BindGroup1::new(&device, &light_ambient);
    let bind_group_view_proj = BindGroup2::new(&device, &view_mat, &proj_mat);

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&surface, &window, &device);

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

                light_direction = rotate_vec3(&light_direction, 0.05);
                let bind_group_light_dir = BindGroup1::new(&device, &light_direction);

                model_mat = rotation_y(model_mat, 0.05);
                let bind_group_model = BindGroup1::new(&device, &model_mat);
                {
                    let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);

                    {
                        let context1 = (&context).set_a_position(&mut rpass, &vertex_position);
                        {
                            let context2 = (&context1).set_a_normal(&mut rpass, &vertex_normal);
                            {
                                let context3 =
                                    (&context2).set_Ambient(&mut rpass, &bind_group_ambient);
                                {
                                    let context4 = (&context3)
                                        .set_u_view_u_proj(&mut rpass, &bind_group_view_proj);
                                    {
                                        let context5 = (&context4)
                                            .set_LightDirection(&mut rpass, &bind_group_light_dir);
                                        {
                                            let context6 = (&context5)
                                                .set_u_model(&mut rpass, &bind_group_model);
                                            {
                                                let _ = context6.runnable(|| {
                                                    graphics_run_indices(rpass, &indices, 1)
                                                });
                                            }
                                        }
                                    }
                                }
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
    /* let window = winit::window::Window::new(&event_loop).unwrap(); */

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
