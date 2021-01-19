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
    default_bind_group, generate_swap_chain, graphics_run_indicies, graphics_starting_context,
    setup_render_pass, setup_render_pass_color_depth, setup_render_pass_depth, BindingPreprocess,
    GraphicsBindings, GraphicsShader, OutGraphicsBindings, PipelineType,
};

pub use pipeline::bind::Bindings;

pub use wgpu_macros::{generic_bindings, init};

pub use pipeline::helper::{
    create_texels, generate_identity_matrix, generate_light_projection, generate_projection_matrix,
    generate_view_matrix, load_cube, load_model, load_plane, rotate_vec4, rotation, scale,
    translate,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    init!();

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

    my_shader! {BAKE_VERTEXT = {
        [[vertex in] vec3] a_position;

        group {
            [[uniform in] mat4] u_viewProj;
            [[uniform in] mat4] u_World;
        }

        [[out] vec4] gl_Position;

        {{
            void main() {
                gl_Position = u_viewProj * u_World * vec4(a_position, 1.0);;
            }
        }}
    }}

    my_shader! {BAKE_FRAGMENT = {
        {{
            void main() {
            }
        }}
    }}

    my_shader! {VERTEXT = {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;

        [[out] vec4] v_Position;
        [[out] vec3] v_Normal;
        [[out] vec4] gl_Position;

        group {
            [[uniform in] mat4] u_viewProj;
            //[[uniform in] mat4] u_proj;
            [[uniform in] mat4] u_World;
        }
        [[uniform in] vec4] u_Color;

        {{
            void main() {
                v_Normal = mat3(u_World) * a_normal;
                v_Position = u_World * vec4(a_position, 1.0);
                gl_Position = u_viewProj * v_Position;
            }
        }}
    }}

    my_shader! {FRAGMENT = {
        [[in] vec3] v_Normal;
        [[in] vec4] v_Position;

        [[out] vec4] color; // This is o_Target in the docs

        [[uniform in] mat4] u_viewProj;
        //[[uniform in] mat4] u_proj;

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
                o_Target += shadow /* * diffuse */ * light_color.xyz;
                // multiply the light by material color
                color = vec4(o_Target, 1.0) * u_Color;
            }
        }}
    }}

    const B_V: GraphicsShader = eager_graphics_shader! {BAKE_VERTEXT!()};
    const B_F: GraphicsShader = eager_graphics_shader! {BAKE_FRAGMENT!()};
    eager_binding! {bake_context = BAKE_VERTEXT!(), BAKE_FRAGMENT!()};

    let (stencil_program, stencil_template_bindings, stencil_template_out_bindings, _) =
        compile_valid_stencil_program!(device, B_V, B_F, PipelineType::Stencil);

    const S_V: GraphicsShader = eager_graphics_shader! {VERTEXT!()};
    const S_F: GraphicsShader = eager_graphics_shader! {FRAGMENT!()};
    eager_binding! {context = VERTEXT!(), FRAGMENT!()};

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(device, S_V, S_F, PipelineType::ColorWithStencil);

    let view_proj_mat =
        generate_projection_matrix(size.width as f32 / size.height as f32) * generate_view_matrix();

    let (positions, normals, index_data) = load_model("src/models/sphere.obj");
    //let (mut positions, mut normals, mut index_data) = load_cube();
    let color_data = vec![[0.583, 0.771, 0.014, 1.0]];
    let world_mat = scale(translate(generate_identity_matrix(), 0.0, 3.0, -1.0), 0.5);

    let (plane_positions, plane_normals, plane_index_data) = load_plane(7);
    let plane_color_data = vec![[1.0, 1.0, 1.0, 1.0]];
    let plane_world_mat = generate_identity_matrix();

    let mut light_pos = vec![[20.0, -30.0, 2.0, 1.0]];
    let mut light_proj_mat = generate_light_projection(light_pos[0], 60.0);
    let light_color = vec![[1.0, 0.5, 0.5, 0.5]];

    let shadow_sampler = wgpu::SamplerDescriptor {
        label: Some("shadow"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare: wgpu::CompareFunction::LessEqual,
        ..Default::default()
    };

    let shadow_size = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth: 1,
    };
    let shadow_format = wgpu::TextureFormat::Depth32Float;

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&surface, &window, &device);

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

    let light_target_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
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

/*
    param1 : in
    param2 : in

    // todo go with this
    // todo fix examples
    // todo fix run
    //
    context2
    context3
    {
        context1 = bind a to param1 in context
        {
            context2 = bind b to param2 in context1
        }
        {
            context3 = bind c to param2 in context1
        }
    }

    render_loop {
        bind light in context2
        bind light in context3

        run(context2)
        run(context3)
    }

    ///////

    program 1
    program 2

    context<Bound, Unbound, ...>
    context<[Bound, Unbound], [Unbound, Unbound], ...>

    bind_param1(data,...)

    bind_param1([Update, Don't Update, ...], data, ...)

    {
        bind a to param1@1@2
        {
            bind b to param2@1
            {
                bind c to param2@2
                render_loop {
                    bind light to paramlight@1@2 {
                        run(context@1)
                        run(context@2)
                    }
                }
            }
        }
    }


    bind a to param1 { // bindgroup = {a}
        render_loop {
            bind b to param2 {
                // bindgroup = {b}
                run()
            }
            bind c to param2 {
                // bindgroup = {c}
                run()
            }
        }
    }
*/

    let mut bind_group = default_bind_group(&device);
    let mut bind_group_p1 = default_bind_group(&device);
    let mut bind_group_stencil = default_bind_group(&device);
    let mut bind_group_stencil_p1 = default_bind_group(&device);

    let mut bindings: GraphicsBindings = template_bindings.new();
    let mut out_bindings: OutGraphicsBindings = template_out_bindings.new();
    let mut bindings_stencil: GraphicsBindings = stencil_template_bindings.new();
    let mut out_bindings_stencil: OutGraphicsBindings = stencil_template_out_bindings.new();

    let bake_context_plane;
    let mut bindings_stencil_plane;
    let mut out_bindings_stencil_plane;

    let bake_context_sphere;
    let mut bindings_stencil_sphere;
    let mut out_bindings_stencil_sphere;

    let context_plane;
    let mut bindings_plane;
    let mut out_bindings_plane;

    let context_sphere;
    let mut bindings_sphere;
    let mut out_bindings_sphere;

    {
        let bake_context1 = (&bake_context).bind_a_position(
            &plane_positions,
            &device,
            &mut bindings_stencil,
            &mut out_bindings_stencil,
        );
        {
            bake_context_plane = (&bake_context1).bind_u_World(
                &plane_world_mat,
                &device,
                &mut bindings_stencil,
                &mut out_bindings_stencil,
            );
            bindings_stencil_plane = bindings_stencil.clone();
            out_bindings_stencil_plane = out_bindings_stencil.clone();
        }
    }

    {
        let bake_context1 = (&bake_context).bind_a_position(
            &positions,
            &device,
            &mut bindings_stencil,
            &mut out_bindings_stencil,
        );
        {
            bake_context_sphere = bake_context1.bind_u_World(
                &world_mat,
                &device,
                &mut bindings_stencil,
                &mut out_bindings_stencil,
            );
            bindings_stencil_sphere = bindings_stencil.clone();
            out_bindings_stencil_sphere = out_bindings_stencil.clone();
        }
    }

    {
        let context1 =
            (&context).bind_u_viewProj(&view_proj_mat, &device, &mut bindings, &mut out_bindings);

        {
            let context2 =
                context1.bind_light_color(&light_color, &device, &mut bindings, &mut out_bindings);

            {
                let context3 = context2.bind_s_Shadow(
                    &shadow_sampler,
                    &device,
                    &mut bindings,
                    &mut out_bindings,
                );

                {
                    let context4 = context3.bind_t_Shadow(
                        &shadow_texture,
                        &device,
                        &mut bindings,
                        &mut out_bindings,
                    );

                    {
                        let context5 = (&context4).bind_a_position(
                            &plane_positions,
                            &device,
                            &mut bindings,
                            &mut out_bindings,
                        );

                        {
                            let context6 = context5.bind_u_Color(
                                &plane_color_data,
                                &device,
                                &mut bindings,
                                &mut out_bindings,
                            );

                            {
                                let context7 = context6.bind_u_World(
                                    &plane_world_mat,
                                    &device,
                                    &mut bindings,
                                    &mut out_bindings,
                                );

                                {
                                    context_plane = context7.bind_a_normal(
                                        &plane_normals,
                                        &device,
                                        &mut bindings,
                                        &mut out_bindings,
                                    );
                                    bindings_plane = bindings.clone();
                                    out_bindings_plane = out_bindings.clone();
                                }
                            }
                        }
                    }
                    {
                        let context5 = (&context4).bind_a_position(
                            &positions,
                            &device,
                            &mut bindings,
                            &mut out_bindings,
                        );
                        {
                            let context6 = context5.bind_u_Color(
                                &color_data,
                                &device,
                                &mut bindings,
                                &mut out_bindings,
                            );

                            {
                                let context7 = context6.bind_u_World(
                                    &world_mat,
                                    &device,
                                    &mut bindings,
                                    &mut out_bindings,
                                );

                                {
                                    context_sphere = context7.bind_a_normal(
                                        &normals,
                                        &device,
                                        &mut bindings,
                                        &mut out_bindings,
                                    );
                                    bindings_sphere = bindings.clone();
                                    out_bindings_sphere = out_bindings.clone();
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    {bind param2
        render_loop {
        bind param1
    run()}
        }

    bind param2 {
        bind param1 {
            run()
        }
    }

    most common -> least common


    non-changing -> changing


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
                    let mut bind_prerun_stencil_p1;
                    let mut bind_prerun_stencil;
                    let mut bind_prerun;
                    let mut bind_prerun_p1;

                    light_pos = rotate_vec4(&light_pos, -0.05);
                    light_proj_mat = generate_light_projection(light_pos[0], 60.0);

                    dbg!(&light_pos);

                    {
                        let mut rpass_stencil = setup_render_pass_depth(
                            &stencil_program,
                            &mut init_encoder,
                            &light_target_view,
                        );

                        {
                            let bake_context3 = (&bake_context_plane).bind_u_viewProj(
                                &light_proj_mat,
                                &device,
                                &mut bindings_stencil_plane,
                                &mut out_bindings_stencil_plane,
                            );

                            {
                                bind_prerun_stencil_p1 = BindingPreprocess::bind(
                                    &mut bindings_stencil_plane,
                                    &out_bindings_stencil_plane,
                                );
                                rpass_stencil = bake_context3.runnable(|| {
                                    graphics_run_indicies(
                                        &stencil_program,
                                        &device,
                                        rpass_stencil,
                                        &mut bind_group_stencil_p1,
                                        &mut bind_prerun_stencil_p1,
                                        &plane_index_data,
                                    )
                                });
                            }
                        }
                        {
                            let bake_context3 = (&bake_context_sphere).bind_u_viewProj(
                                &light_proj_mat,
                                &device,
                                &mut bindings_stencil_sphere,
                                &mut out_bindings_stencil_sphere,
                            );
                            {
                                bind_prerun_stencil = BindingPreprocess::bind(
                                    &mut bindings_stencil_sphere,
                                    &out_bindings_stencil_sphere,
                                );
                                rpass_stencil = bake_context3.runnable(|| {
                                    graphics_run_indicies(
                                        &stencil_program,
                                        &device,
                                        rpass_stencil,
                                        &mut bind_group_stencil,
                                        &mut bind_prerun_stencil,
                                        &index_data,
                                    )
                                });
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

                        {
                            let context9 = (&context_plane).bind_light_proj(
                                &light_proj_mat,
                                &device,
                                &mut bindings_plane,
                                &mut out_bindings_plane,
                            );

                            {
                                let context10 = context9.bind_light_pos(
                                    &light_pos,
                                    &device,
                                    &mut bindings_plane,
                                    &mut out_bindings_plane,
                                );
                                {
                                    bind_prerun_p1 = BindingPreprocess::bind(
                                        &mut bindings_plane,
                                        &out_bindings_plane,
                                    );
                                    rpass = context10.runnable(|| {
                                        graphics_run_indicies(
                                            &program,
                                            &device,
                                            rpass,
                                            &mut bind_group_p1,
                                            &mut bind_prerun_p1,
                                            &plane_index_data,
                                        )
                                    });
                                }
                            }
                        }

                        {
                            let context9 = (&context_sphere).bind_light_proj(
                                &light_proj_mat,
                                &device,
                                &mut bindings_sphere,
                                &mut out_bindings_sphere,
                            );

                            {
                                let context10 = context9.bind_light_pos(
                                    &light_pos,
                                    &device,
                                    &mut bindings_sphere,
                                    &mut out_bindings_sphere,
                                );
                                {
                                    bind_prerun = BindingPreprocess::bind(
                                        &mut bindings_sphere,
                                        &out_bindings_sphere,
                                    );
                                    rpass = context10.runnable(|| {
                                        graphics_run_indicies(
                                            &program,
                                            &device,
                                            rpass,
                                            &mut bind_group,
                                            &mut bind_prerun,
                                            &index_data,
                                        )
                                    });
                                }
                            }
                        }
                    }

                    queue.submit(&[init_encoder.finish()]);
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

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
