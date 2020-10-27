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
    load_cube, load_plane, translate,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const BAKE_VERTEXT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[vertex in] vec3] a_position;

        [[uniform in] mat4] u_viewProj;
        [[uniform in] mat4] u_World;
        [[uniform in] vec4] u_Color;

        [[out] vec4] gl_Position;


        {{
            void main() {
                gl_Position = u_viewProj * u_World * vec4(a_position, 1.0);
            }
        }}
    };

    const BAKE_FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        {{
            void main() {}
        }}
    };

    const VERTEXT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;

        [[out] vec4] v_Position;
        [[out] vec3] v_Normal;
        [[out] vec4] gl_Position;

        [[uniform in] mat4] u_viewProj;
        [[uniform in] mat4] u_World;
        [[uniform in] vec4] u_Color;

        {{
            void main() {
                v_Normal = mat3(u_World) * a_normal;
                v_Position = u_World * vec4(a_position, 1.0);
                gl_Position = u_viewProj * v_Position;
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, BindingContext) = graphics_shader! {
        [[in] vec3] v_Normal;
        [[in] vec4] v_Position;

        [[out] vec4] color; // This is o_Target in the docs

        [[uniform in] mat4] u_viewProj;

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
        compile_valid_stencil_program!(window, B_V, B_F, PipelineType::Stencil);

    const S_V: GraphicsShader = VERTEXT.0;
    const S_F: GraphicsShader = FRAGMENT.0;
    const VERTEXT_STARTING_BIND_CONTEXT: BindingContext = VERTEXT.1;
    const STARTING_BIND_CONTEXT: BindingContext =
        graphics_starting_context(VERTEXT_STARTING_BIND_CONTEXT, S_F);

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(window, S_V, S_F, PipelineType::ColorWithStencil);

    let (positions, normals, index_data) = load_cube();

    let (plane_positions, plane_normals, plane_index_data) = load_plane(7);

    //todo should I create a bunch of entity cubes?

    let color_data = vec![[0.583, 0.771, 0.014, 1.0]];

    let view_proj_mat =
        generate_view_matrix() * generate_projection_matrix(size.width as f32 / size.height as f32);

    let world_mat = generate_identity_matrix();

    let light_proj_mat_init =
        generate_view_matrix() * generate_projection_matrix(size.width as f32 / size.height as f32);
    let light_proj_mat = translate(light_proj_mat_init, 2.0, 0.5, 0.0);

    let light_pos = vec![[7.0, -5.0, 10.0, 1.0]];
    let light_color = vec![[0.5, 1.0, 0.5, 1.0]];

    //let model_mat2 = translate(model_mat, 2.0, 0.0, 0.0);

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
                {
                    let mut bindings: GraphicsBindings = template_bindings.clone();
                    let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();
                    let mut bind_group = default_bind_group(&program);

                    let mut bindings_stencil: GraphicsBindings = stencil_template_bindings.clone();
                    let mut out_bindings_stencil: OutGraphicsBindings =
                        stencil_template_out_bindings.clone();
                    let mut bind_group_stencil = default_bind_group(&stencil_program);
                    let mut bind_prerun_stencil;

                    let mut bind_prerun;
                    /* let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame); */

                    let context = Context::new();

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

                    let tex_size = 1600u32;
                    let texels = create_texels(tex_size as usize);

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
                    let shadow_texture = program.device.create_texture(&wgpu::TextureDescriptor {
                        size: shadow_size,
                        array_layer_count: 1, // One light
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
                            base_array_layer: 0,
                            array_layer_count: 1,
                        });

                    let depth_texture = program.device.create_texture(&wgpu::TextureDescriptor {
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

                    {
                        {
                            let mut rpass_stencil = setup_render_pass_depth(
                                &stencil_program,
                                &mut init_encoder,
                                &light_target_view,
                            );

                            const BAKE_BIND_CONTEXT_1: BindingContext =
                                update_bind_context(&BAKE_STARTING_BIND_CONTEXT, "a_position");
                            let context1 = bind!(
                                stencil_program,
                                bindings_stencil,
                                out_bindings_stencil,
                                "a_position",
                                positions,
                                context,
                                BAKE_BIND_CONTEXT_1
                            );
                            {
                                const BAKE_BIND_CONTEXT_2: BindingContext =
                                    update_bind_context(&BAKE_BIND_CONTEXT_1, "u_viewProj");
                                let context2 = bind!(
                                    stencil_program,
                                    bindings_stencil,
                                    out_bindings_stencil,
                                    "u_viewProj",
                                    view_proj_mat,
                                    context1,
                                    BAKE_BIND_CONTEXT_2
                                );
                                {
                                    const BAKE_BIND_CONTEXT_3: BindingContext =
                                        update_bind_context(&BAKE_BIND_CONTEXT_2, "u_Color");
                                    let context3 = bind!(
                                        stencil_program,
                                        bindings_stencil,
                                        out_bindings_stencil,
                                        "u_Color",
                                        color_data,
                                        context2,
                                        BAKE_BIND_CONTEXT_3
                                    );
                                    {
                                        const BAKE_BIND_CONTEXT_4: BindingContext =
                                            update_bind_context(&BAKE_BIND_CONTEXT_3, "u_World");
                                        let context4 = bind!(
                                            stencil_program,
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
                                            dbg!();
                                            rpass_stencil = graphics_run_indicies(
                                                &stencil_program,
                                                rpass_stencil,
                                                &mut bind_group_stencil,
                                                &mut bind_prerun_stencil,
                                                &index_data,
                                            );
                                            dbg!();
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
                            const BIND_CONTEXT_1: BindingContext =
                                update_bind_context(&STARTING_BIND_CONTEXT, "a_position");
                            let context1 = bind!(
                                program,
                                bindings,
                                out_bindings,
                                "a_position",
                                positions,
                                context,
                                BIND_CONTEXT_1
                            );
                            {
                                const BIND_CONTEXT_2: BindingContext =
                                    update_bind_context(&BIND_CONTEXT_1, "u_viewProj");
                                let context2 = bind!(
                                    program,
                                    bindings,
                                    out_bindings,
                                    "u_viewProj",
                                    view_proj_mat,
                                    context1,
                                    BIND_CONTEXT_2
                                );
                                {
                                    const BIND_CONTEXT_3: BindingContext =
                                        update_bind_context(&BIND_CONTEXT_2, "u_Color");
                                    let context3 = bind!(
                                        program,
                                        bindings,
                                        out_bindings,
                                        "u_Color",
                                        color_data,
                                        context2,
                                        BIND_CONTEXT_3
                                    );
                                    {
                                        const BIND_CONTEXT_4: BindingContext =
                                            update_bind_context(&BIND_CONTEXT_3, "u_World");
                                        let context4 = bind!(
                                            program,
                                            bindings,
                                            out_bindings,
                                            "u_World",
                                            world_mat,
                                            context3,
                                            BIND_CONTEXT_4
                                        );

                                        {
                                            const BIND_CONTEXT_5: BindingContext =
                                                update_bind_context(&BIND_CONTEXT_4, "a_normal");
                                            let context5 = bind!(
                                                program,
                                                bindings,
                                                out_bindings,
                                                "a_normal",
                                                normals,
                                                context4,
                                                BIND_CONTEXT_5
                                            );
                                            {
                                                const BIND_CONTEXT_6: BindingContext =
                                                    update_bind_context(
                                                        &BIND_CONTEXT_5,
                                                        "light_proj",
                                                    );
                                                let context6 = bind!(
                                                    program,
                                                    bindings,
                                                    out_bindings,
                                                    "light_proj",
                                                    light_proj_mat,
                                                    context5,
                                                    BIND_CONTEXT_6
                                                );

                                                {
                                                    const BIND_CONTEXT_7: BindingContext =
                                                        update_bind_context(
                                                            &BIND_CONTEXT_6,
                                                            "light_pos",
                                                        );
                                                    let context7 = bind!(
                                                        program,
                                                        bindings,
                                                        out_bindings,
                                                        "light_pos",
                                                        light_pos,
                                                        context6,
                                                        BIND_CONTEXT_7
                                                    );
                                                    {
                                                        const BIND_CONTEXT_8: BindingContext =
                                                            update_bind_context(
                                                                &BIND_CONTEXT_7,
                                                                "light_color",
                                                            );
                                                        let context8 = bind!(
                                                            program,
                                                            bindings,
                                                            out_bindings,
                                                            "light_color",
                                                            light_color,
                                                            context7,
                                                            BIND_CONTEXT_8
                                                        );
                                                        {
                                                            const BIND_CONTEXT_9: BindingContext =
                                                                update_bind_context(
                                                                    &BIND_CONTEXT_8,
                                                                    "s_Shadow",
                                                                );
                                                            bind_sampler(
                                                                &program,
                                                                &mut bindings,
                                                                &mut out_bindings,
                                                                sampler,
                                                                "s_Shadow".to_string(),
                                                            );
                                                            dbg!(&bindings);
                                                            {
                                                                const BIND_CONTEXT_10:
                                                                    BindingContext =
                                                                    update_bind_context(
                                                                        &BIND_CONTEXT_9,
                                                                        "t_Shadow",
                                                                    );
                                                                bind_texture(
                                                                    &program,
                                                                    &mut bindings,
                                                                    &mut out_bindings,
                                                                    shadow_view,
                                                                    "t_Shadow".to_string(),
                                                                );
                                                                {
                                                                    const _: () = ready_to_run(
                                                                        BIND_CONTEXT_10,
                                                                    );
                                                                    bind_prerun =
                                                                        BindingPreprocess::bind(
                                                                            &mut bindings,
                                                                            &out_bindings,
                                                                        );
                                                                    dbg!();
                                                                    rpass = graphics_run_indicies(
                                                                        &program,
                                                                        rpass,
                                                                        &mut bind_group,
                                                                        &mut bind_prerun,
                                                                        &index_data,
                                                                    );
                                                                    dbg!();
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                dbg!();
                            }
                        }
                        dbg!();
                    }
                }
                dbg!();
                program.queue.submit(&[init_encoder.finish()]);
                dbg!();
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
