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
    bind_sampler, bind_texture, compile_buffer, graphics_starting_context, valid_fragment_shader,
    valid_vertex_shader, GraphicsBindings, GraphicsShader, OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec2, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run,
    Bindings,
};

pub use pipeline::helper::load_cube;

pub use pipeline::context::BindingContext;

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const FORWARD_VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec4] a_Pos;
        [[vertex in] vec4] a_Normal;
        [[out] vec3] v_Normal;
        [[out] vec4] v_Position;
        [[out] vec4] gl_Position;

        [[uniform in] mat4] u_ViewProj;
        /* [[uniform in] vec4] u_NumLights; */

        [[uniform in] mat4] u_World;

        {{
            void main() {
                v_Normal = mat3(u_World) * vec3(a_Normal.xyz);
                v_Position = u_World * vec4(a_Pos);
                gl_Position = u_ViewProj * v_Position;
            }
        }}
    };

    const FORWARD_FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[in] vec3] v_Normal;
        [[in] vec4] v_Position;
        [[out] vec4] color;

        [[uniform in] mat4] uViewProj;
        // We are starting with just 1 light

        [[uniform in] mat4] Light_proj;
        [[uniform in] vec4] Light_pos;
        [[uniform in] vec4] Light_color;

        /*             [[uniform in] texture2DArray] t_Shadow;
        [[uniform in] samplerShadow] s_Shadow; */

        [[uniform in] mat4] u_World;
        [[uniform in] vec4] u_Color;

        /*         float fetch_shadow(int light_id, vec4 homogeneous_coords) {
            if (homogeneous_coords.w <= 0.0) {
                return 1.0;
            }
            // compensate for the Y-flip difference between the NDC and texture coordinates
            const vec2 flip_correction = vec2(0.5, -0.5);
            // compute texture coordinates for shadow lookup
            vec4 light_local = vec4(
                homogeneous_coords.xy * flip_correction/homogeneous_coords.w + 0.5,
                light_id,
                homogeneous_coords.z / homogeneous_coords.w
            );
            // do the lookup, using HW PCF and comparison
            return texture(sampler2DArrayShadow(t_Shadow, s_Shadow), light_local);
        } */

        {{
            void main() {
                vec3 normal = normalize(v_Normal);
                vec3 ambient = vec3(0.05, 0.05, 0.05);
                // accumulate color
                vec3 f_color = ambient;
                /*                 for (int i=0; i<int(u_NumLights.x) && i<MAX_LIGHTS; ++i) {
                    Light light = u_Lights[i];
                    // project into the light space
                    float shadow = fetch_shadow(i, light.proj * v_Position);
                    // compute Lambertian diffuse term
                    vec3 light_dir = normalize(light.pos.xyz - v_Position.xyz);
                    float diffuse = max(0.0, dot(normal, light_dir));
                    // add light contribution
                    f_color += shadow * diffuse * light.color.xyz;
                } */

                // multiply the light by material color
                color = vec4(f_color, 1.0) * u_Color;
            }
        }}
    };

    const S_v: GraphicsShader = FORWARD_VERTEXT.0;
    const VERTEX_STARTING_BIND_CONTEXT: [&str; 32] = FORWARD_VERTEXT.1;
    const S_f: GraphicsShader = FORWARD_FRAGMENT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] =
        graphics_starting_context(VERTEX_STARTING_BIND_CONTEXT, S_f);

    println!("{:?}", STARTING_BIND_CONTEXT);
    panic!("JUST CHECKING");

    let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    static_assertions::const_assert!(valid_vertex_shader(&S_v));
    static_assertions::const_assert!(valid_fragment_shader(&S_f));
    let (program, mut template_bindings, mut template_out_bindings) =
        wgpu_graphics_header::graphics_compile(&mut compile_buffer, &window, &S_v, &S_f).await;

    let (positions, normals, indicies) = load_cube();

    fn generate_transform(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );

        mx_correction * mx_projection * mx_view
    }

    // For drawing to window
    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        // Only update during the "vertical blanking interval"
        // As opposed to Immediate where it is possible to see visual tearing(where multiple frames are visible at once)
        present_mode: wgpu::PresentMode::Immediate,
    };

    let trans_mat = generate_transform(sc_desc.width as f32 / sc_desc.height as f32);

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

                /*                 let multisampled_texture_extent = wgpu::Extent3d {
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
                */

                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();

                /*                 let sampler = program.device.create_sampler(&wgpu::SamplerDescriptor {
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
                */

                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "a_Pos");
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &positions,
                    "a_Pos".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1, "a_Normal");
                    bind_vec3(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        &normals,
                        "a_Normal".to_string(),
                    );
                    {
                        const BIND_CONTEXT_2: [&str; 32] =
                            update_bind_context!(BIND_CONTEXT_1, "u_Transform");
                        bind_mat4(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            *i,
                            "u_Transform".to_string(),
                        );
                        {
                            const BIND_CONTEXT_3: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_2, "a_TexCoord");
                            bind_vec2(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                &texture_coordinates,
                                "a_TexCoord".to_string(),
                            );
                            {
                                const BIND_CONTEXT_4: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_3, "t_Color");
                                bind_texture(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    texture_view,
                                    "t_Color".to_string(),
                                );
                                {
                                    /* const BIND_CONTEXT_5: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_4, "s_Color"); */
                                    bind_sampler(
                                        &program,
                                        &mut bindings,
                                        &mut out_bindings,
                                        sampler,
                                        "s_Color".to_string(),
                                    );
                                    {
                                        /* ready_to_run(BIND_CONTEXT_5); */
                                        if *i == trans_mat {
                                            wgpu_graphics_header::graphics_multi_run_indicies(
                                                &program,
                                                &mut init_encoder,
                                                &mut bindings,
                                                out_bindings,
                                                &mut frame,
                                                &mut multi_sampled_view,
                                                &index_data,
                                            )
                                        } else {
                                            wgpu_graphics_header::graphics_run_indicies(
                                                &program,
                                                &mut init_encoder,
                                                &mut bindings,
                                                out_bindings,
                                                &mut frame,
                                                &index_data,
                                            )
                                        };
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
