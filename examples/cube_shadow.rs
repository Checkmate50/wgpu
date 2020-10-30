#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header::{
    bind_sampler, bind_texture, default_bind_group, generate_swap_chain, graphics_run_indicies,
    graphics_starting_context, setup_render_pass, setup_render_pass_color_depth,
    setup_render_pass_depth, BindingPreprocess, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings, PipelineType,
};

pub use pipeline::shared::{is_gl_builtin, Bindable, Bindings, Context};

pub use pipeline::context::{ready_to_run, update_bind_context, BindingContext};

pub use pipeline::helper::{
    create_texels, generate_identity_matrix, generate_projection_matrix, generate_view_matrix,
    load_cube, load_model, load_plane, rotation, scale, translate, generate_light_projection, rotate_vec4
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    // Create a surface to draw images on
    let surface = wgpu::Surface::create(&window);
    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            // Can specify Low/High power usage
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        },
        // Map to Vulkan/Metal/Direct3D 12
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    // The device manages the connection and resources of the adapter
    // The queue is a literal queue of tasks for the gpu
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        })
        .await;

    const BAKE_VERTEXT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[vertex in] vec3] a_position;

        [[uniform in] mat4] u_viewProj;
        [[uniform in] mat4] u_World;

        [[out] vec4] gl_Position;

        {{
            void main() {
                gl_Position = u_viewProj * u_World * vec4(a_position, 1.0);;
            }
        }}
    };

    const BAKE_FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        {{
            void main() {
            }
        }}
    };

    const VERTEXT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;

        [[out] vec4] v_Position;
        [[out] vec3] v_Normal;
        [[out] vec4] gl_Position;

        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;
        [[uniform in] mat4] u_World;
        [[uniform in] vec4] u_Color;

        {{
            void main() {
                v_Normal = mat3(u_World) * a_normal;
                v_Position = u_World * vec4(a_position, 1.0);
                gl_Position = u_proj * u_view * v_Position;
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[in] vec3] v_Normal;
        [[in] vec4] v_Position;

        [[out] vec4] color; // This is o_Target in the docs

        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;

        // We are starting with just one light
        [[uniform in] mat4] light_proj;
        [[uniform in] vec4] light_pos;
        [[uniform in] vec4] light_color;

        [[uniform in] texture2DArray] t_Shadow;
        [[uniform in] samplerShadow] s_Shadow;
        [[uniform in] mat4] u_World;
        [[uniform in] vec4] u_Color;
        {{
            float fetch_shadow(int light_id, vec4 homogeneous_coords) {
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
            }

            void main() {
                vec3 normal = normalize(v_Normal);
                // accumulate color
                vec3 ambient = vec3(0.05, 0.05, 0.05);
                vec3 o_Target = ambient;

                // project into the light space
                float shadow = fetch_shadow(0, light_proj * v_Position);
                // compute Lambertian diffuse term

                vec3 light_dir = normalize(light_pos.xyz - v_Position.xyz);
                float diffuse = max(0.0, dot(normal, light_dir));
                // add light contribution
                o_Target += shadow * diffuse * light_color.xyz;
                // multiply the light by material color
                color = vec4(o_Target, 1.0) * u_Color;
            }
        }}
    };

    const B_V: GraphicsShader = BAKE_VERTEXT.0;
    const B_F: GraphicsShader = BAKE_FRAGMENT.0;
    const BAKE_STARTING_BIND_CONTEXT: BindingContext = BAKE_VERTEXT.1;

    let (stencil_program, stencil_template_bindings, stencil_template_out_bindings, _) =
        compile_valid_stencil_program!(device, B_V, B_F, PipelineType::Stencil);

    const S_V: GraphicsShader = VERTEXT.0;
    const S_F: GraphicsShader = FRAGMENT.0;
    const VERTEXT_STARTING_BIND_CONTEXT: BindingContext = VERTEXT.1;
    const STARTING_BIND_CONTEXT: BindingContext =
        graphics_starting_context(VERTEXT_STARTING_BIND_CONTEXT, S_F);

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(device, S_V, S_F, PipelineType::ColorWithStencil);

    let view_mat = generate_view_matrix();
    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);

    let (mut positions, mut normals, mut index_data) = load_model("src/models/sphere.obj");
    //let (mut positions, mut normals, mut index_data) = load_cube();
    let color_data = vec![[0.583, 0.771, 0.014, 1.0]];
    let world_mat = scale(translate(generate_identity_matrix(), 0.0, 3.0, -1.0), 0.5);

    let (mut plane_positions, mut plane_normals, mut plane_index_data) = load_plane(7);
    let plane_color_data = vec![[1.0, 1.0, 1.0, 1.0]];
    let plane_world_mat = generate_identity_matrix();


    let mut light_pos = vec![[20.0, -30.0, 2.0, 1.0]];
    let mut light_proj_mat = generate_light_projection(light_pos[0], 60.0);
    let light_color = vec![[1.0, 0.5, 0.5, 0.5]];

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
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");
                {
                    let mut bindings: GraphicsBindings = template_bindings.clone();
                    let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();
                    let mut bind_group = default_bind_group(&device);
                    let mut bind_group_p1 = default_bind_group(&device);

                    light_pos = rotate_vec4(&light_pos, -0.05);
                    light_proj_mat = generate_light_projection(light_pos[0], 60.0);

                    dbg!(&light_pos);

                    let mut bindings_stencil: GraphicsBindings = stencil_template_bindings.clone();
                    let mut out_bindings_stencil: OutGraphicsBindings =
                        stencil_template_out_bindings.clone();
                    let mut bind_group_stencil = default_bind_group(&device);
                    let mut bind_group_stencil_p1 = default_bind_group(&device);

                    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        address_mode_u: wgpu::AddressMode::ClampToEdge,
                        address_mode_v: wgpu::AddressMode::ClampToEdge,
                        address_mode_w: wgpu::AddressMode::ClampToEdge,
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        mipmap_filter: wgpu::FilterMode::Nearest,
                        lod_min_clamp: -100.0,
                        lod_max_clamp: 100.0,
                        compare: wgpu::CompareFunction::LessEqual,
                    });

                    let tex_size = 1600u32;

                    let shadow_size = wgpu::Extent3d {
                        width: 512,
                        height: 512,
                        depth: 1,
                    };
                    let shadow_format = wgpu::TextureFormat::Depth32Float;
                    let texture_extent = wgpu::Extent3d {
                        width: tex_size,
                        height: tex_size,
                        depth: 1,
                    };
                    let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
                        size: shadow_size,
                        array_layer_count: 1, // One light (max lights)
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: shadow_format,
                        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
                        label: None,
                    });
                    let shadow_view = shadow_texture.create_default_view();

                    let light_target_view =
                        shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                            format: shadow_format,
                            dimension: wgpu::TextureViewDimension::D2,
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0, // The first light is at index 0
                            array_layer_count: 1,
                        });

                    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: window.inner_size().width,
                            height: window.inner_size().height,
                            depth: 1,
                        },
                        array_layer_count: 1,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                        label: None,
                    });
                    let depth_view = depth_texture.create_default_view();

                    let mut bind_prerun_stencil_p1;
                    let mut bind_prerun_stencil;
                    let mut bind_prerun;
                    let mut bind_prerun_p1;

                    let context = Context::new();
                    {
                        {
                            let mut rpass_stencil = setup_render_pass_depth(
                                &stencil_program,
                                &mut init_encoder,
                                &light_target_view,
                            );

                            {
                            const BAKE_BIND_CONTEXT_1: BindingContext =
                                update_bind_context(&BAKE_STARTING_BIND_CONTEXT, "u_viewProj");
                            let context1 = bind!(
                                device,
                                bindings_stencil,
                                out_bindings_stencil,
                                "u_viewProj",
                                light_proj_mat,
                                context,
                                BAKE_BIND_CONTEXT_1
                            );
                            /* const BAKE_BIND_CONTEXT_2: BindingContext =
                                update_bind_context(&BAKE_BIND_CONTEXT_1, "u_proj");
                            let context2 = bind!(
                                device,
                                bindings_stencil,
                                out_bindings_stencil,
                                "u_proj",
                                proj_mat,
                                context1,
                                BAKE_BIND_CONTEXT_2
                            ); */
                                {
                                const BAKE_BIND_CONTEXT_3: BindingContext =
                                    update_bind_context(&BAKE_BIND_CONTEXT_1, "a_position");
                                let context3 = bind!(
                                    device,
                                    bindings_stencil,
                                    out_bindings_stencil,
                                    "a_position",
                                    plane_positions,
                                    context1,
                                    BAKE_BIND_CONTEXT_3
                                );
                                    {
                                        const BAKE_BIND_CONTEXT_4: BindingContext =
                                            update_bind_context(&BAKE_BIND_CONTEXT_3, "u_World");
                                        let context4 = bind!(
                                            device,
                                            bindings_stencil,
                                            out_bindings_stencil,
                                            "u_World",
                                            plane_world_mat,
                                            context3,
                                            BAKE_BIND_CONTEXT_4
                                        );
                                        {
                                            const _: () = ready_to_run(BAKE_BIND_CONTEXT_4);
                                            bind_prerun_stencil_p1 = BindingPreprocess::bind(
                                                &mut bindings_stencil,
                                                &out_bindings_stencil,
                                            );
                                            rpass_stencil = graphics_run_indicies(
                                                &stencil_program,
                                                &device,
                                                rpass_stencil,
                                                &mut bind_group_stencil_p1,
                                                &mut bind_prerun_stencil_p1,
                                                &plane_index_data,
                                            );
                                        }
                                    }
                                }

                            {
                                const BAKE_BIND_CONTEXT_3: BindingContext =
                                    update_bind_context(&BAKE_BIND_CONTEXT_1, "a_position");
                                let context3 = bind!(
                                    device,
                                    bindings_stencil,
                                    out_bindings_stencil,
                                    "a_position",
                                    positions,
                                    context1,
                                    BAKE_BIND_CONTEXT_3
                                );
                                    {
                                        const BAKE_BIND_CONTEXT_4: BindingContext =
                                            update_bind_context(&BAKE_BIND_CONTEXT_3, "u_World");
                                        let context4 = bind!(
                                            device,
                                            bindings_stencil,
                                            out_bindings_stencil,
                                            "u_World",
                                            world_mat,
                                            context3,
                                            BAKE_BIND_CONTEXT_4
                                        );
                                        {
                                            const _: () = ready_to_run(BAKE_BIND_CONTEXT_4);
                                            bind_prerun_stencil = BindingPreprocess::bind(
                                                &mut bindings_stencil,
                                                &out_bindings_stencil,
                                            );
                                            rpass_stencil = graphics_run_indicies(
                                                &stencil_program,
                                                &device,
                                                rpass_stencil,
                                                &mut bind_group_stencil,
                                                &mut bind_prerun_stencil,
                                                &index_data,
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        {
                            let mut rpass = setup_render_pass_color_depth(
                                &program,
                                &mut init_encoder,
                                &frame,
                                &depth_view,
                            );
                            /* let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame); */

                            const BIND_CONTEXT_1: BindingContext =
                                update_bind_context(&STARTING_BIND_CONTEXT, "u_view");
                            let context1 = bind!(
                                device,
                                bindings,
                                out_bindings,
                                "u_view",
                                view_mat,
                                context,
                                BIND_CONTEXT_1
                            );
                            const BIND_CONTEXT_2: BindingContext =
                                update_bind_context(&BIND_CONTEXT_1, "u_proj");
                            let context2 = bind!(
                                device,
                                bindings,
                                out_bindings,
                                "u_proj",
                                proj_mat,
                                context1,
                                BIND_CONTEXT_2
                            );
                            const BIND_CONTEXT_3: BindingContext =
                                update_bind_context(&BIND_CONTEXT_2, "light_proj");
                            let context3 = bind!(
                                device,
                                bindings,
                                out_bindings,
                                "light_proj",
                                light_proj_mat,
                                context2,
                                BIND_CONTEXT_3
                            );

                            const BIND_CONTEXT_4: BindingContext =
                                update_bind_context(&BIND_CONTEXT_3, "light_pos");
                            let context4 = bind!(
                                device,
                                bindings,
                                out_bindings,
                                "light_pos",
                                light_pos,
                                context3,
                                BIND_CONTEXT_4
                            );

                            const BIND_CONTEXT_5: BindingContext =
                                update_bind_context(&BIND_CONTEXT_4, "light_color");
                            let context5 = bind!(
                                device,
                                bindings,
                                out_bindings,
                                "light_color",
                                light_color,
                                context4,
                                BIND_CONTEXT_5
                            );

                            const BIND_CONTEXT_6: BindingContext =
                                update_bind_context(
                                    &BIND_CONTEXT_5,
                                    "s_Shadow",
                                );
                            bind_sampler(
                                &mut bindings,
                                &mut out_bindings,
                                shadow_sampler,
                                "s_Shadow".to_string(),
                            );

                            const BIND_CONTEXT_7:
                                BindingContext =
                                update_bind_context(
                                    &BIND_CONTEXT_6,
                                    "t_Shadow",
                                );
                            bind_texture(
                                &mut bindings,
                                &mut out_bindings,
                                shadow_view,
                                "t_Shadow".to_string(),
                            );

                            {
                                const BIND_CONTEXT_8_P1: BindingContext =
                                    update_bind_context(&BIND_CONTEXT_7, "a_position");
                                let context8_P1 = bind!(
                                    device,
                                    bindings,
                                    out_bindings,
                                    "a_position",
                                    plane_positions,
                                    context5,
                                    BIND_CONTEXT_8_P1
                                );
                                {
                                    const BIND_CONTEXT_9_P1: BindingContext =
                                        update_bind_context(&BIND_CONTEXT_8_P1, "u_Color");
                                    let context9_P1 = bind!(
                                        device,
                                        bindings,
                                        out_bindings,
                                        "u_Color",
                                        plane_color_data,
                                        context8_P1,
                                        BIND_CONTEXT_9_P1
                                    );
                                    {
                                        const BIND_CONTEXT_10_P1: BindingContext =
                                            update_bind_context(&BIND_CONTEXT_9_P1, "u_World");
                                        let context10_P1 = bind!(
                                            device,
                                            bindings,
                                            out_bindings,
                                            "u_World",
                                            plane_world_mat,
                                            context9_P1,
                                            BIND_CONTEXT_10_P1
                                        );

                                        {
                                            const BIND_CONTEXT_11_P1: BindingContext =
                                                update_bind_context(
                                                    &BIND_CONTEXT_10_P1,
                                                    "a_normal",
                                                );
                                            let context11_P1 = bind!(
                                                device,
                                                bindings,
                                                out_bindings,
                                                "a_normal",
                                                plane_normals,
                                                context10_P1,
                                                BIND_CONTEXT_11_P1
                                            );
                                            {
                                                const _: () = ready_to_run(BIND_CONTEXT_11_P1);
                                                bind_prerun_p1 = BindingPreprocess::bind(
                                                    &mut bindings,
                                                    &out_bindings,
                                                );
                                                rpass = graphics_run_indicies(
                                                    &program,
                                                    &device,
                                                    rpass,
                                                    &mut bind_group_p1,
                                                    &mut bind_prerun_p1,
                                                    &plane_index_data,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            {
                                const BIND_CONTEXT_8: BindingContext =
                                    update_bind_context(&BIND_CONTEXT_7, "a_position");
                                let context8 = bind!(
                                    device,
                                    bindings,
                                    out_bindings,
                                    "a_position",
                                    positions,
                                    context5,
                                    BIND_CONTEXT_8
                                );
                                {
                                    const BIND_CONTEXT_9: BindingContext =
                                        update_bind_context(&BIND_CONTEXT_8, "u_Color");
                                    let context9 = bind!(
                                        device,
                                        bindings,
                                        out_bindings,
                                        "u_Color",
                                        color_data,
                                        context8,
                                        BIND_CONTEXT_9
                                    );
                                    {
                                        const BIND_CONTEXT_10: BindingContext =
                                            update_bind_context(&BIND_CONTEXT_9, "u_World");
                                        let context10 = bind!(
                                            device,
                                            bindings,
                                            out_bindings,
                                            "u_World",
                                            world_mat,
                                            context9,
                                            BIND_CONTEXT_10
                                        );

                                        {
                                            const BIND_CONTEXT_11: BindingContext =
                                                update_bind_context(&BIND_CONTEXT_10, "a_normal");
                                            let context11 = bind!(
                                                device,
                                                bindings,
                                                out_bindings,
                                                "a_normal",
                                                normals,
                                                context10,
                                                BIND_CONTEXT_11
                                            );
                                            {
                                                const _: () = ready_to_run(BIND_CONTEXT_11);
                                                bind_prerun = BindingPreprocess::bind(
                                                    &mut bindings,
                                                    &out_bindings,
                                                );
                                                rpass = graphics_run_indicies(
                                                    &program,
                                                    &device,
                                                    rpass,
                                                    &mut bind_group,
                                                    &mut bind_prerun,
                                                    &index_data,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        queue.submit(&[init_encoder.finish()]);
                    }
                }
                /* std::process::exit(0); */
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
