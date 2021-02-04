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
    compile_buffer, default_bind_group, generate_swap_chain, setup_render_pass,
    valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec3, is_gl_builtin, ready_to_run, update_bind_context, Bindings,
};

pub use pipeline::helper::load_cube;

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] vertexColor;
        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;


        [[out] vec3] fragmentColor;
        [[out] vec4] gl_Position;
        {{
            void main() {
                fragmentColor = vertexColor;
                gl_Position = u_proj * u_view * vec4(0.3 * a_position, 1.0);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[in] vec3] fragmentColor;
        [[out] vec4] color;
        {{
            void main() {
                color = vec4(fragmentColor, 1.0);
            }
        }}
    };

    const S_V: GraphicsShader = VERTEXT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(window, S_V, S_F);

    let (positions, _, index_data) = load_cube();

    let color_data = vec![
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
    ];

    let light_direction = vec![[20.0, 0.0, 20.0]];

    let light_ambient = vec![[0.1, 0.0, 0.0]];

    fn generate_view(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        mx_view
    }

    fn generate_projection(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );

        mx_correction * mx_projection
    }

    let view_mat = generate_view(size.width as f32 / size.height as f32);

    let proj_mat = generate_projection(size.width as f32 / size.height as f32);

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&program, &window);

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
                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context(&STARTING_BIND_CONTEXT, "a_position");
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &positions,
                    "a_position".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context(&BIND_CONTEXT_1, "u_view");
                    bind_mat4(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        view_mat,
                        "u_view".to_string(),
                    );
                    {
                        const BIND_CONTEXT_3: [&str; 32] =
                            update_bind_context(&BIND_CONTEXT_2, "vertexColor");
                        bind_vec3(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            &color_data,
                            "vertexColor".to_string(),
                        );

                        {
                            const BIND_CONTEXT_4: [&str; 32] =
                                update_bind_context(&BIND_CONTEXT_3, "u_proj");
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                /*                             const BIND_CONTEXT_5: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_4, "vertexAmbient");
                                bind_vec3(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    &light_ambient,
                                    "vertexAmbient".to_string(),
                                ); */
                                {
                                    /*                                     const BIND_CONTEXT_6: [&str; 32] = update_bind_context!(
                                        BIND_CONTEXT_5,
                                        "vertexLightDirection"
                                    );
                                    bind_vec3(
                                        &program,
                                        &mut bindings,
                                        &mut out_bindings,
                                        &light_direction,
                                        "vertexLightDirection".to_string(),
                                    ); */
                                    {
                                        ready_to_run(BIND_CONTEXT_4);
                                        wgpu_graphics_header::graphics_run_indicies(
                                            &program,
                                            rpass,
                                            &mut bind_group,
                                            &mut bindings,
                                            &mut out_bindings,
                                            &index_data,
                                        );
                                    }
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