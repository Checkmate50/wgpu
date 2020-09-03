#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header::{
    default_bind_group, generate_swap_chain, graphics_run_indicies, setup_render_pass,
    GraphicsBindings, GraphicsShader, OutGraphicsBindings,
};

pub use pipeline::shared::{bind_fvec, bind_mat4, bind_vec3, Bindings};

pub use pipeline::context::{ready_to_run, update_bind_context, BindingContext, MetaContext};

pub use pipeline::helper::{
    generate_identity_matrix, generate_projection_matrix, generate_view_matrix, load_model,
    rotation_y,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;
        [[uniform in] vec3] Ambient;
        [[uniform in] vec3] LightDirection;
        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;
        [[uniform in] mat4] u_model;
        [[] int] gl_VertexID;

        [[out] vec3] fragmentNormal;
        [[out] vec4] gl_Position;
        {{
            void main() {

                vec4 worldNormal = vec4(a_normal, 0.0) * inverse(u_model) * inverse(u_view);

                fragmentNormal = worldNormal.xyz;

                gl_Position = u_proj * u_view * u_model * vec4(0.7 * a_position, 1.0);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[in] vec3] fragmentNormal;
        [[uniform in] vec3] Ambient;
        [[uniform in] vec3] LightDirection;
        [[out] vec4] color;
        {{
            void main() {
                vec3 fragColor = vec3(1.0, 0.0, 0.0);
                color = vec4(Ambient + fragColor * max(dot(normalize(fragmentNormal), normalize(LightDirection)), 0.0), 1.0);
            }
        }}
    };

    const S_V: GraphicsShader = VERTEXT.0;
    const STARTING_BIND_CONTEXT: BindingContext = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;
    const STARTING_META_CONTEXT: MetaContext = MetaContext::new();

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(window, S_V, S_F);

    let (positions, normals, indices) = load_model("src/models/teapot.obj");

    let mut light_direction = vec![[20.0, 0.0, 0.0]];

    let light_ambient = vec![[0.1, 0.0, 0.0]];

    let view_mat = generate_view_matrix();

    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);

    let mut model_mat = generate_identity_matrix();

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

                let frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");

                let rpass = setup_render_pass(&program, &mut init_encoder, &frame);
                let mut bind_group = default_bind_group(&program);

                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();

                model_mat = rotation_y(model_mat, 0.05);

                fn rotate_vec3(start: &Vec<[f32; 3]>, delta_y: f32) -> Vec<[f32; 3]> {
                    let mut temp_vec3 = cgmath::Vector3::new(start[0][0], start[0][1], start[0][2]);
                    temp_vec3 = cgmath::Matrix3::from_angle_y(cgmath::Rad(delta_y)) * temp_vec3;
                    vec![[temp_vec3.x, temp_vec3.y, temp_vec3.z]]
                };

                light_direction = rotate_vec3(&light_direction, 0.05);

                const BIND_CONTEXT_1: (BindingContext, MetaContext) = update_bind_context(
                    &STARTING_BIND_CONTEXT,
                    "a_position",
                    STARTING_META_CONTEXT,
                    "BIND_CONTEXT_1",
                );
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &positions,
                    "a_position".to_string(),
                );
                {
                    const BIND_CONTEXT_2: (BindingContext, MetaContext) = update_bind_context(
                        &BIND_CONTEXT_1.0,
                        "u_view",
                        BIND_CONTEXT_1.1,
                        "BIND_CONTEXT_2",
                    );
                    bind_mat4(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        view_mat,
                        "u_view".to_string(),
                    );
                    {
                        const BIND_CONTEXT_3: (BindingContext, MetaContext) = update_bind_context(
                            &BIND_CONTEXT_2.0,
                            "u_model",
                            BIND_CONTEXT_2.1,
                            "BIND_CONTEXT_3",
                        );
                        bind_mat4(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            model_mat,
                            "u_model".to_string(),
                        );
                        {
                            const BIND_CONTEXT_4: (BindingContext, MetaContext) =
                                update_bind_context(
                                    &BIND_CONTEXT_3.0,
                                    "u_proj",
                                    BIND_CONTEXT_3.1,
                                    "BIND_CONTEXT_4",
                                );
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                const BIND_CONTEXT_5: (BindingContext, MetaContext) =
                                    update_bind_context(
                                        &BIND_CONTEXT_4.0,
                                        "Ambient",
                                        BIND_CONTEXT_4.1,
                                        "BIND_CONTEXT_5",
                                    );
                                bind_vec3(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    &light_ambient,
                                    "Ambient".to_string(),
                                );
                                {
                                    const BIND_CONTEXT_6: (BindingContext, MetaContext) =
                                        update_bind_context(
                                            &BIND_CONTEXT_5.0,
                                            "LightDirection",
                                            BIND_CONTEXT_5.1,
                                            "BIND_CONTEXT_6",
                                        );
                                    bind_vec3(
                                        &program,
                                        &mut bindings,
                                        &mut out_bindings,
                                        &light_direction,
                                        "LightDirection".to_string(),
                                    );
                                    {
                                        const BIND_CONTEXT_7: (BindingContext, MetaContext) =
                                            update_bind_context(
                                                &BIND_CONTEXT_6.0,
                                                "a_normal",
                                                BIND_CONTEXT_6.1,
                                                "BIND_CONTEXT_7",
                                            );
                                        bind_vec3(
                                            &program,
                                            &mut bindings,
                                            &mut out_bindings,
                                            &normals,
                                            "a_normal".to_string(),
                                        );
                                        {
                                            const Next_Meta_Context: MetaContext =
                                                ready_to_run(BIND_CONTEXT_7.0, BIND_CONTEXT_7.1);
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
