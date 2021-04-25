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

pub use pipeline::bind::{BindGroup1, BindGroup2, Indices, SamplerData, TextureData, Vertex};
pub use pipeline::AbstractBind;

pub use pipeline::helper::{
    create_texels, generate_identity_matrix, generate_light_projection, generate_projection_matrix,
    generate_view_matrix, load_cube, load_model, load_plane, rotate_vec4, rotation, scale,
    translate,
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
                limits: wgpu::Limits {
                    max_bind_groups: 5,
                    ..Default::default()
                },
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let queue = std::rc::Rc::new(queue);

    my_shader! {BAKE_VERTEXT = {
        [[vertex in] vec3] a_position;

        [group1 [uniform in] mat4] u_viewProj;
        [group1 [uniform in] mat4] u_World;

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


        [group1 [uniform in] mat4] u_viewProj;
        //[[uniform in] mat4] u_proj;
        [group1 [uniform in] mat4] u_World;

        [group2 [uniform in] vec4] u_Color;

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

        [group1 [uniform in] mat4] u_viewProj;
        //[[uniform in] mat4] u_proj;

        // We are starting with just one light
        [group4 [uniform in] mat4] light_proj;
        [group4 [uniform in] vec4] light_pos;
        [group5 [uniform in] vec4] light_color;

        [group3 [uniform in] texture2DArray] t_Shadow;
        [group3 [uniform in compare] samplerShadow] s_Shadow;
        [group1 [uniform in] mat4] u_World;
        [group2 [uniform in] vec4] u_Color;
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
    }}

    const B_V: GraphicsShader = eager_graphics_shader! {BAKE_VERTEXT!()};
    const B_F: GraphicsShader = eager_graphics_shader! {BAKE_FRAGMENT!()};
    eager_binding! {bake_context = BAKE_VERTEXT!(), BAKE_FRAGMENT!()};

    let (stencil_program, _) = compile_valid_stencil_program!(
        device,
        bake_context,
        B_V,
        B_F,
        GraphicsCompileArgs {
            primitive_state: wgpu::PrimitiveState {
                cull_mode: wgpu::CullMode::Back,
                ..Default::default()
            },
            color_target_state: None,
            depth_stencil_state: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
                clamp_depth: device.features().contains(wgpu::Features::DEPTH_CLAMPING),
            }),
            multisample_state: wgpu::MultisampleState::default(),
        }
    );

    const S_V: GraphicsShader = eager_graphics_shader! {VERTEXT!()};
    const S_F: GraphicsShader = eager_graphics_shader! {FRAGMENT!()};
    eager_binding! {context = VERTEXT!(), FRAGMENT!()};

    let args = GraphicsCompileArgs {
        primitive_state: wgpu::PrimitiveState {
            cull_mode: wgpu::CullMode::Back,
            ..Default::default()
        },
        color_target_state: Some(wgpu::ColorTargetState {
            // Specify the size of the color data in the buffer
            // Bgra8UnormSrgb is specifically used since it is guaranteed to work on basically all browsers (32bit)
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            // Here is where you can do some fancy stuff for transitioning colors/brightness between frames. Replace defaults to taking all of the current frame and none of the next frame.
            // This can be changed by specifying the modifier for either of the values from src/dest frames or changing the operation used to combine them(instead of addition maybe Max/Min)
            color_blend: wgpu::BlendState::REPLACE,
            alpha_blend: wgpu::BlendState::REPLACE,
            // We can adjust the mask to only include certain colors if we want to
            write_mask: wgpu::ColorWrite::ALL,
        }),
        depth_stencil_state: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
            clamp_depth: false,
        }),
        multisample_state: wgpu::MultisampleState::default(),
    };

    let (program, _) = compile_valid_graphics_program!(device, context, S_V, S_F, args);

    let view_proj_mat_init =
        generate_projection_matrix(size.width as f32 / size.height as f32) * generate_view_matrix();
    let view_proj_mat = rotation(view_proj_mat_init, 0.0, 0.0, 0.0);
    let world_mat = scale(translate(generate_identity_matrix(), 1.0, 3.0, -1.0), 0.5);

    let bind_group_view_world = BindGroup2::new(&device, &view_proj_mat, &world_mat);

    let (positions_data, normals_data, index_data) = load_model("src/models/sphere.obj");

    let positions = Vertex::new(&device, &positions_data);
    let normals = Vertex::new(&device, &normals_data);
    let index = Indices::new(&device, &index_data);

    //let (mut positions, mut normals, mut index_data) = load_cube();
    let color_data = vec![[0.583, 0.771, 0.014, 1.0]];
    let color = BindGroup1::new(&device, &color_data);

    let (plane_positions_data, plane_normals_data, plane_index_data) = load_plane(7);

    let plane_positions = Vertex::new(&device, &plane_positions_data);
    let plane_normals = Vertex::new(&device, &plane_normals_data);
    let plane_index = Indices::new(&device, &plane_index_data);

    let plane_color_data = vec![[1.0, 1.0, 1.0, 1.0]];
    let plane_color = BindGroup1::new(&device, &plane_color_data);

    let plane_world_mat = generate_identity_matrix();
    let bind_group_plane_world_mat = BindGroup2::new(&device, &view_proj_mat, &plane_world_mat);

    let mut light_pos = vec![[20.0, -30.0, 2.0, 1.0]];
    //let mut light_proj_mat = generate_light_projection(light_pos[0], 60.0);
    let light_color_data = vec![[1.0, 0.5, 0.5, 0.5]];
    let light_color = BindGroup1::new(&device, &light_color_data);

    let shadow_sampler = SamplerData::new(wgpu::SamplerDescriptor {
        label: Some("shadow"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    // A "chain" of buffers that we render on to the display
    let swap_chain = generate_swap_chain(&surface, &window, &device);

    let shadow_texture = TextureData::new_without_data(
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
            label: None,
        },
        wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: std::num::NonZeroU32::new(1),
            base_array_layer: 0, // The first light is at index 0
            array_layer_count: std::num::NonZeroU32::new(1),
        },
        queue.clone(),
    );

    let shadow_t_s = BindGroup2::new(&device, &shadow_texture, &shadow_sampler);

    let depth_texture = BindGroup1::<
        TextureData<
            { pipeline::bind::TextureMultisampled::False },
            { wgpu::TextureSampleType::Depth },
            { wgpu::TextureViewDimension::D2 },
        >,
    >::new(
        &device,
        &TextureData::new_without_data(
            wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: window.inner_size().width,
                    height: window.inner_size().height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
                label: None,
            },
            wgpu::TextureViewDescriptor::default(),
            queue.clone(),
        ),
    );

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
                    light_pos = rotate_vec4(&light_pos, -0.05);
                    let light_proj_mat = generate_light_projection(light_pos[0], 60.0);

                    let bind_group_light_proj_pos =
                        BindGroup2::new(&device, &light_proj_mat, &light_pos);

                    //dbg!(&light_pos);

                    {
                        let shadow_view = shadow_t_s.get_view_0(&wgpu::TextureViewDescriptor {
                            label: None,
                            format: Some(wgpu::TextureFormat::Depth32Float),
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            level_count: std::num::NonZeroU32::new(1),
                            base_array_layer: 0, // The first light is at index 0
                            array_layer_count: std::num::NonZeroU32::new(1),
                        });
                        let mut rpass_stencil = setup_render_pass(
                            &stencil_program,
                            &mut init_encoder,
                            wgpu::RenderPassDescriptor {
                                label: None,
                                // color_attachments is literally where we draw the colors to
                                color_attachments: &[],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                        attachment: &shadow_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: true,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                            },
                        );
                        {
                            let bake_context1 = (&bake_context)
                                .set_a_position(&mut rpass_stencil, &plane_positions);
                            {
                                let bake_context_plane = (&bake_context1).set_u_viewProj_u_World(
                                    &mut rpass_stencil,
                                    &bind_group_plane_world_mat,
                                );
                                {
                                    rpass_stencil = bake_context_plane.runnable(|| {
                                        graphics_run_indices(rpass_stencil, &plane_index, 1)
                                    });
                                }
                            }
                        }

                        {
                            let bake_context1 =
                                (&bake_context).set_a_position(&mut rpass_stencil, &positions);
                            {
                                let bake_context_sphere = bake_context1.set_u_viewProj_u_World(
                                    &mut rpass_stencil,
                                    &bind_group_view_world,
                                );

                                {
                                    let _ = bake_context_sphere.runnable(|| {
                                        graphics_run_indices(rpass_stencil, &index, 1)
                                    });
                                }
                            }
                        }
                    }

                    {
                        let depth_view =
                            depth_texture.get_view_0(&wgpu::TextureViewDescriptor::default());
                        let mut rpass = setup_render_pass(
                            &program,
                            &mut init_encoder,
                            wgpu::RenderPassDescriptor {
                                label: None,
                                // color_attachments is literally where we draw the colors to
                                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                    attachment: &frame.view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.1,
                                            g: 0.2,
                                            b: 0.3,
                                            a: 1.0,
                                        }),
                                        store: true,
                                    },
                                }],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                        attachment: &depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: true,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                            },
                        );

                        {
                            let context1 = (&context)
                                .set_u_viewProj_u_World(&mut rpass, &bind_group_view_world);

                            {
                                let context2 = context1.set_light_color(&mut rpass, &light_color);

                                {
                                    let context3 =
                                        context2.set_t_Shadow_s_Shadow(&mut rpass, &shadow_t_s);
                                    //shadow_texture

                                    {
                                        let context5 = (&context3)
                                            .set_a_position(&mut rpass, &plane_positions);

                                        {
                                            let context6 =
                                                context5.set_u_Color(&mut rpass, &plane_color);

                                            {
                                                let context_plane = context6
                                                    .set_a_normal(&mut rpass, &plane_normals);
                                                {
                                                    let context9 = (&context_plane)
                                                        .set_light_proj_light_pos(
                                                            &mut rpass,
                                                            &bind_group_light_proj_pos,
                                                        );

                                                    {
                                                        rpass = context9.runnable(|| {
                                                            graphics_run_indices(
                                                                rpass,
                                                                &plane_index,
                                                                1,
                                                            )
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    {
                                        let context5 =
                                            (&context3).set_a_position(&mut rpass, &positions);
                                        {
                                            let context6 = context5.set_u_Color(&mut rpass, &color);

                                            {
                                                let context_sphere =
                                                    context6.set_a_normal(&mut rpass, &normals);
                                                {
                                                    let context9 = (&context_sphere)
                                                        .set_light_proj_light_pos(
                                                            &mut rpass,
                                                            &bind_group_light_proj_pos,
                                                        );

                                                    {
                                                        let _ = context9.runnable(|| {
                                                            graphics_run_indices(rpass, &index, 1)
                                                        });
                                                    }
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
