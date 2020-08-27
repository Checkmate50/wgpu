#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use static_assertions::const_assert;

pub use pipeline::wgpu_graphics_header;
pub use pipeline::wgpu_graphics_header::{
    compile_buffer, default_bind_group, graphics_run_indicies, setup_render_pass,
    valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run, Bindings,
};

pub use pipeline::helper::{
    generate_identity_matrix, generate_projection_matrix, generate_view_matrix, load_model,
    rotation_x, rotation_y, rotation_z, scale, translate,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;
        [[uniform in] mat4] u_model;

        [[out] vec4] gl_Position;
        {{
            void main() {
                gl_Position = u_proj * u_view * u_model * vec4( a_position, 1.0);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[out] vec4] color;
        {{
            void main() {
                color = vec4(1.0, 0.4, 0.25, 1.0);
            }
        }}
    };

    const S_V: GraphicsShader = VERTEXT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;

    let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    static_assertions::const_assert!(valid_vertex_shader(&S_V));
    static_assertions::const_assert!(valid_fragment_shader(&S_F));
    let (program, template_bindings, template_out_bindings) =
        wgpu_graphics_header::graphics_compile(&mut compile_buffer, &window, &S_V, &S_F).await;

    let (positions, normals, indices) = load_model("src/models/teapot.obj");

    /*     println!("{:?}", positions); */

    // For drawing to window
    let sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        // Only update during the "vertical blanking interval"
        // As opposed to Immediate where it is possible to see visual tearing(where multiple frames are visible at once)
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut view_mat = generate_view_matrix();
    /* view_mat = translate(view_mat, 0.0, -1.0, 0.1); */

    /*     println!("{:?}", generate_view_matrix()); */

    println!("View mat: {:?}", view_mat);

    let mut proj_mat = generate_projection_matrix(sc_desc.width as f32 / sc_desc.height as f32);

    /*     proj_mat = rotation_x(proj_mat, 34.0);
    proj_mat = rotation_y(proj_mat, 3.0); */

    let mut model_mat = generate_identity_matrix();
    /*     model_mat = scale(model_mat, 0.5); */

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = program.device.create_swap_chain(&program.surface, &sc_desc);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut init_encoder = program
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                let mut frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");

                let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);
                let mut bind_group = default_bind_group(&program);

                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();

                model_mat = rotation_y(model_mat, 0.05);

                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "a_position");
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &positions,
                    "a_position".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1, "u_view");
                    bind_mat4(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        view_mat,
                        "u_view".to_string(),
                    );
                    {
                        {
                            const BIND_CONTEXT_3: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_2, "u_proj");
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                const BIND_CONTEXT_4: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_3, "u_model");
                                bind_mat4(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    model_mat,
                                    "u_model".to_string(),
                                );
                                {
                                    ready_to_run(BIND_CONTEXT_4);
                                    graphics_run_indicies(
                                        &program,
                                        rpass,
                                        &mut bind_group,
                                        &mut bindings,
                                        &out_bindings,
                                        &indices,
                                    );
                                }
                            }
                        }
                    }
                }
                program.queue.submit(&[init_encoder.finish()]);
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
