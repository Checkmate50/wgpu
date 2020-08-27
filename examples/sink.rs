// Everything but the kitchen sink
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
    bind_sampler, bind_texture, compile_buffer, default_bind_group, setup_render_pass,
    valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings, graphics_starting_context
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run, Bindings, bind_vec2
};

pub use pipeline::helper::{
    generate_identity_matrix, generate_projection_matrix, generate_view_matrix, load_model,
    rotation_x, rotation_y, rotation_z, translate, create_texels, load_cube
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;
        [[uniform in] vec3] Ambient;
        [[uniform in] vec3] LightDirection;
        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;
        [[uniform in] mat4] u_model;

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

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
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
    const STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;

    let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    static_assertions::const_assert!(valid_vertex_shader(&S_V));
    static_assertions::const_assert!(valid_fragment_shader(&S_F));
    let (program, template_bindings, template_out_bindings) =
        wgpu_graphics_header::graphics_compile(&mut compile_buffer, &window, &S_V, &S_F).await;

    const VERTEXT_CUBE: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec3] a_Pos;
        [[vertex in] vec2] a_TexCoord;
        [[uniform in] mat4] u_Transform;

        [[out] vec2] v_TexCoord;
        [[out] vec4] gl_Position;
        {{
            void main() {
                v_TexCoord = a_TexCoord;
                gl_Position = u_Transform * vec4(a_Pos, 1.0);
            }
        }}
    };

    const FRAGMENT_CUBE: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[in] vec2] v_TexCoord;
        [[out] vec4] color;
        [[uniform in] texture2D] t_Color;
        [[uniform in] sampler] s_Color;
        {{
            void main() {
                vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
                float mag = length(v_TexCoord-vec2(0.5));
                color = mix(tex, vec4(0.0), mag*mag);
            }
        }}
    };

    const S_v_CUBE: GraphicsShader = VERTEXT_CUBE.0;
    const VERTEXT_STARTING_BIND_CONTEXT_CUBE: [&str; 32] = VERTEXT_CUBE.1;
    const S_f_CUBE: GraphicsShader = FRAGMENT_CUBE.0;
    const STARTING_BIND_CONTEXT_CUBE: [&str; 32] =
        graphics_starting_context(VERTEXT_STARTING_BIND_CONTEXT_CUBE, S_f_CUBE);


    //let mut compile_buffer2: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    static_assertions::const_assert!(valid_vertex_shader(&S_v_CUBE));
    static_assertions::const_assert!(valid_fragment_shader(&S_f_CUBE));
    let (program_CUBE, mut template_bindings_CUBE, mut template_out_bindings_CUBE) =
        wgpu_graphics_header::graphics_compile(&mut compile_buffer, &window, &S_v_CUBE, &S_f_CUBE).await;

    let (positions, normals, indices) = load_model("src/models/teapot.obj");

    let (positions2, normals2, indices2) = load_model("src/models/caiman.obj");

    let (positions_cube, normals_cube, index_data_cube) = load_cube();

    let texture_coordinates_cube: Vec<[f32; 2]> = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];

    let mut light_direction = vec![[20.0, 0.0, 0.0]];

    let light_ambient = vec![[0.1, 0.0, 0.0]];

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

    let view_mat = generate_view_matrix();

    let proj_mat = generate_projection_matrix(sc_desc.width as f32 / sc_desc.height as f32);

    let mut model_mat = generate_identity_matrix();

    let mut model_mat2 = rotation_x(translate(model_mat, 0.5, -3.0, 2.0), 2.0);

    let mut model_mat3 = translate(model_mat, 0.5, 0.0, -0.5);

    // rust is going the reverse of the order we want for matrix multiplication
    let trans_mat = model_mat3 * proj_mat * view_mat;

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

                let mut bind_group = default_bind_group(&program);
                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();

                let mut bind_group2 = default_bind_group(&program);
                let mut bindings2: GraphicsBindings = template_bindings.clone();
                let mut out_bindings2: OutGraphicsBindings = template_out_bindings.clone();

                let mut bind_group3 = default_bind_group(&program_CUBE);
                let mut bindings3: GraphicsBindings = template_bindings_CUBE.clone();
                let mut out_bindings3: OutGraphicsBindings = template_out_bindings_CUBE.clone();

                let multisampled_texture_extent = wgpu::Extent3d {
                    width: sc_desc.width,
                    height: sc_desc.height,
                    depth: 1,
                };
                let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
                    size: multisampled_texture_extent,
                    array_layer_count: 1,
                    mip_level_count: 1,
                    sample_count: 4,
                    dimension: wgpu::TextureDimension::D2,
                    format: sc_desc.format,
                    usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                    label: None,
                };

                let mut multi_sampled_view = program
                    .device
                    .create_texture(multisampled_frame_descriptor)
                    .create_default_view();

                let sampler = program.device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    lod_min_clamp: -100.0,
                    lod_max_clamp: 100.0,
                    compare: wgpu::CompareFunction::Undefined,
                });

                let tex_size = 256u32;
                let texels = create_texels(tex_size as usize);
                let texture_extent = wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth: 1,
                };
                let texture = program.device.create_texture(&wgpu::TextureDescriptor {
                    size: texture_extent,
                    array_layer_count: 1,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                    label: None,
                });
                let texture_view = texture.create_default_view();
                let temp_buf = program
                    .device
                    .create_buffer_with_data(texels.as_slice(), wgpu::BufferUsage::COPY_SRC);
                init_encoder.copy_buffer_to_texture(
                    wgpu::BufferCopyView {
                        buffer: &temp_buf,
                        offset: 0,
                        bytes_per_row: 4 * tex_size,
                        rows_per_image: 0,
                    },
                    wgpu::TextureCopyView {
                        texture: &texture,
                        mip_level: 0,
                        array_layer: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    texture_extent,
                );

                let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);

                model_mat = rotation_y(model_mat, 0.05);

                fn rotate_vec3(start: &Vec<[f32; 3]>, delta_y: f32) -> Vec<[f32; 3]> {
                    let mut temp_vec3 = cgmath::Vector3::new(start[0][0], start[0][1], start[0][2]);
                    temp_vec3 = cgmath::Matrix3::from_angle_y(cgmath::Rad(delta_y)) * temp_vec3;
                    vec![[temp_vec3.x, temp_vec3.y, temp_vec3.z]]
                };

                light_direction = rotate_vec3(&light_direction, 0.05);

                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "a_normal");
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &normals,
                    "a_normal".to_string(),
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
                        const BIND_CONTEXT_3: [&str; 32] =
                            update_bind_context!(BIND_CONTEXT_2, "u_model");
                        bind_mat4(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            model_mat,
                            "u_model".to_string(),
                        );
                        {
                            const BIND_CONTEXT_4: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_3, "u_proj");
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                const BIND_CONTEXT_5: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_4, "Ambient");
                                bind_vec3(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    &light_ambient,
                                    "Ambient".to_string(),
                                );
                                {
                                    const BIND_CONTEXT_6: [&str; 32] =
                                        update_bind_context!(BIND_CONTEXT_5, "LightDirection");
                                    bind_vec3(
                                        &program,
                                        &mut bindings,
                                        &mut out_bindings,
                                        &light_direction,
                                        "LightDirection".to_string(),
                                    );
                                    {
                                        const BIND_CONTEXT_7: [&str; 32] =
                                            update_bind_context!(BIND_CONTEXT_6, "a_position");

                                        bind_vec3(
                                            &program,
                                            &mut bindings,
                                            &mut out_bindings,
                                            &positions,
                                            "a_position".to_string(),
                                        );
                                        {
                                            ready_to_run(BIND_CONTEXT_7);
                                            rpass = wgpu_graphics_header::graphics_run_indicies(
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
                const BIND_CONTEXT_1_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "a_normal");
                bind_vec3(
                    &program,
                    &mut bindings2,
                    &mut out_bindings2,
                    &normals2,
                    "a_normal".to_string(),
                );
                {
                    const BIND_CONTEXT_2_1: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1_1, "u_view");
                    bind_mat4(
                        &program,
                        &mut bindings2,
                        &mut out_bindings2,
                        view_mat,
                        "u_view".to_string(),
                    );
                    {
                        const BIND_CONTEXT_3_1: [&str; 32] =
                            update_bind_context!(BIND_CONTEXT_2_1, "u_model");
                        bind_mat4(
                            &program,
                            &mut bindings2,
                            &mut out_bindings2,
                            model_mat2,
                            "u_model".to_string(),
                        );
                        {
                            const BIND_CONTEXT_4_1: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_3_1, "u_proj");
                            bind_mat4(
                                &program,
                                &mut bindings2,
                                &mut out_bindings2,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                const BIND_CONTEXT_5_1: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_4_1, "Ambient");
                                bind_vec3(
                                    &program,
                                    &mut bindings2,
                                    &mut out_bindings2,
                                    &light_ambient,
                                    "Ambient".to_string(),
                                );
                                {
                                    const BIND_CONTEXT_6_1: [&str; 32] =
                                        update_bind_context!(BIND_CONTEXT_5_1, "LightDirection");
                                    bind_vec3(
                                        &program,
                                        &mut bindings2,
                                        &mut out_bindings2,
                                        &light_direction,
                                        "LightDirection".to_string(),
                                    );
                                    {
                                        const BIND_CONTEXT_7_1: [&str; 32] =
                                            update_bind_context!(BIND_CONTEXT_6_1, "a_position");

                                        bind_vec3(
                                            &program,
                                            &mut bindings2,
                                            &mut out_bindings2,
                                            &positions2,
                                            "a_position".to_string(),
                                        );
                                        {
                                            ready_to_run(BIND_CONTEXT_7_1);
                                            rpass = wgpu_graphics_header::graphics_run_indicies(
                                                &program,
                                                rpass,
                                                &mut bind_group2,
                                                &mut bindings2,
                                                &out_bindings2,
                                                &indices2,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                rpass.set_pipeline(&program_CUBE.pipeline);

                const BIND_CONTEXT_1_2: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT_CUBE, "a_Pos");
                bind_vec3(
                    &program_CUBE,
                    &mut bindings3,
                    &mut out_bindings3,
                    &positions_cube,
                    "a_Pos".to_string(),
                );
                {
                    const BIND_CONTEXT_2_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1_2, "u_Transform");
                    bind_mat4(
                        &program_CUBE,
                        &mut bindings3,
                        &mut out_bindings3,
                        trans_mat,
                        "u_Transform".to_string(),
                    );
                    {
                        const BIND_CONTEXT_3_2: [&str; 32] =
                            update_bind_context!(BIND_CONTEXT_2_2, "a_TexCoord");
                        bind_vec2(
                            &program_CUBE,
                            &mut bindings3,
                            &mut out_bindings3,
                            &texture_coordinates_cube,
                            "a_TexCoord".to_string(),
                        );
                        {
                            const BIND_CONTEXT_4_2: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_3_2, "t_Color");
                            bind_texture(
                                &program_CUBE,
                                &mut bindings3,
                                &mut out_bindings3,
                                texture_view,
                                "t_Color".to_string(),
                            );
                            {
                                const BIND_CONTEXT_5_2: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_4_2, "s_Color");
                                bind_sampler(
                                    &program_CUBE,
                                    &mut bindings3,
                                    &mut out_bindings3,
                                    sampler,
                                    "s_Color".to_string(),
                                );
                                {
                                    static_assertions::const_assert!(ready_to_run(
                                        BIND_CONTEXT_5_2
                                    ));
                                    wgpu_graphics_header::graphics_run_indicies(
                                        &program_CUBE,
                                        rpass,
                                        &mut bind_group3,
                                        &mut bindings3,
                                        &out_bindings3,
                                        &index_data_cube,
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
