// Everything but the kitchen sink
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

pub use pipeline::bind::{BindGroup1, BindGroup2, Indices, SamplerData, TextureData, Vertex, BufferData};
pub use pipeline::AbstractBind;

pub use pipeline::helper::{
    create_texels, generate_identity_matrix, generate_projection_matrix, generate_view_matrix,
    load_cube, load_model, rotate_vec3, rotation_x, rotation_y, rotation_z, translate,
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

    my_shader! {VERTEX = {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;
        [group1 [uniform in] vec3] Ambient;
        [group2 [uniform in] vec3] LightDirection;
        [group3 [uniform in] mat4] u_view;
        [group3 [uniform in] mat4] u_proj;
        [group4 [uniform in] mat4] u_model;

        [[out] vec3] fragmentNormal;
        [[out] vec4] gl_Position;
        {{
            void main() {

                vec4 worldNormal = vec4(a_normal, 0.0) * inverse(u_model) * inverse(u_view);

                fragmentNormal = worldNormal.xyz;

                gl_Position = u_proj * u_view * u_model * vec4(0.7 * a_position, 1.0);
            }
        }}
    }}

    my_shader! { FRAGMENT = {
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

    const S_V: GraphicsShader = eager_graphics_shader! {VERTEX!()};
    const S_F: GraphicsShader = eager_graphics_shader! {FRAGMENT!()};
    eager_binding! {context = VERTEX!(), FRAGMENT!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, S_V, S_F, GraphicsCompileArgs::default());

    my_shader! { VERTEX_CUBE = {
        [[vertex in] vec3] a_Pos;
        [[vertex in] vec2] a_TexCoord;
        [group1 [uniform in] mat4] u_Transform;

        [[out] vec2] v_TexCoord;
        [[out] vec4] gl_Position;
        {{
            void main() {
                v_TexCoord = a_TexCoord;
                gl_Position = u_Transform * vec4(a_Pos, 1.0);
            }
        }}
    }}

    my_shader! { FRAGMENT_CUBE = {
        [[in] vec2] v_TexCoord;
        [[out] vec4] color;
        [group2 [uniform in] texture2D] t_Color;
        [group2 [uniform in] sampler] s_Color;
        {{
            void main() {
                vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
                float mag = length(v_TexCoord-vec2(0.5));
                color = mix(tex, vec4(0.0), mag*mag);
            }
        }}
    }}

    const S_V_CUBE: GraphicsShader = eager_graphics_shader! {VERTEX_CUBE!()};
    const S_F_CUBE: GraphicsShader = eager_graphics_shader! {FRAGMENT_CUBE!()};
    eager_binding! {context_cube = VERTEX_CUBE!(), FRAGMENT_CUBE!()};

    let (program_CUBE, _) = compile_valid_graphics_program!(
        device,
        context_cube,
        S_V_CUBE,
        S_F_CUBE,
        GraphicsCompileArgs::default()
    );

    let (positions_data, normals_data, indices_data) = load_model("src/models/teapot.obj");
    let positions = Vertex::new(&device, &BufferData::new(positions_data));
    let normals = Vertex::new(&device, &BufferData::new(normals_data));
    let indices = Indices::new(&device, &indices_data);

    let (positions2_data, normals2_data, indices2_data) = load_model("src/models/caiman.obj");
    let positions2 = Vertex::new(&device, &BufferData::new(positions2_data));
    let normals2 = Vertex::new(&device, &BufferData::new(normals2_data));
    let indices2 = Indices::new(&device, &indices2_data);

    let (positions_cube_data, normals_cube_data, index_cube_data) = load_cube();
    let positions_cube = Vertex::new(&device, &BufferData::new(positions_cube_data));
    //let normals_cube = Vertex::new(&device, &BufferData::new(normals_cube_data));
    let index_cube = Indices::new(&device, &index_cube_data);

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
    let texture_coords = Vertex::new(&device, &BufferData::new(texture_coordinates_cube));

    let mut light_direction = vec![[20.0, 0.0, 0.0]];

    let light_ambient_data = vec![[0.1, 0.0, 0.0]];
    let light_ambient = BindGroup1::new(&device, &BufferData::new(light_ambient_data));

    let view_mat = generate_view_matrix();

    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);

    let bg_view_proj = BindGroup2::new(&device, &BufferData::new(view_mat), &BufferData::new(proj_mat));

    let mut model_mat_data = generate_identity_matrix();

    let model_mat2_data = rotation_x(translate(model_mat_data, 0.5, -3.0, 2.0), 2.0);
    let model_mat2 = BindGroup1::new(&device, &BufferData::new(model_mat2_data));

    let model_mat3_data = translate(model_mat_data, 0.5, 0.0, -0.5);
    //let model_mat3 = BindGroup1::new(&device, &BufferData::new(model_mat3_data));

    // rust is going the reverse of the order we want for matrix multiplication
    let trans_mat_data = model_mat3_data * proj_mat * view_mat;
    let trans_mat = BindGroup1::new(&device, &BufferData::new(trans_mat_data));

    let sampler = SamplerData::new(wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare: None,
        ..Default::default()
    });

    let texture = TextureData::new(
        create_texels(256u32 as usize),
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 256u32,
                height: 256u32,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        },
        wgpu::TextureViewDescriptor::default(),
        queue.clone(),
    );

    let bind_group_t_s = BindGroup2::new(&device, &texture, &sampler);

    // A "chain" of buffers that we render on to the display
    let swap_chain = generate_swap_chain(&surface, &window, &device);

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

                model_mat_data = rotation_y(model_mat_data, 0.05);
                let model_mat = BindGroup1::new(&device, &BufferData::new(model_mat_data));

                light_direction = rotate_vec3(&light_direction, 0.05);
                let light_dir = BindGroup1::new(&device, &BufferData::new(light_direction.clone()));

                {
                    let mut rpass = setup_render_pass(
                        &program,
                        &mut init_encoder,
                        wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: true,
                                },
                            }],
                            depth_stencil_attachment: None,
                        },
                    );
                    {
                        let context1 = (&context).set_u_view_u_proj(&mut rpass, &bg_view_proj);

                        {
                            let context2 = context1.set_Ambient(&mut rpass, &light_ambient);
                            {
                                let context3 = context2.set_LightDirection(&mut rpass, &light_dir);
                                {
                                    let context4 = (&context3).set_a_normal(&mut rpass, &normals);
                                    {
                                        let context5 = context4.set_u_model(&mut rpass, &model_mat);
                                        {
                                            let context6 =
                                                context5.set_a_position(&mut rpass, &positions);
                                            {
                                                context6.runnable(|| {
                                                    graphics_run_indices(&mut rpass, &indices, 1)
                                                });
                                            }
                                        }
                                    }
                                }
                                {
                                    let context4 = (&context3).set_a_normal(&mut rpass, &normals2);
                                    {
                                        let context5 =
                                            context4.set_u_model(&mut rpass, &model_mat2);
                                        {
                                            let context6 =
                                                context5.set_a_position(&mut rpass, &positions2);
                                            {
                                                context6.runnable(|| {
                                                    graphics_run_indices(&mut rpass, &indices2, 1)
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    rpass.set_pipeline(&program_CUBE.pipeline);
                    /* let mut rpass = setup_render_pass(
                        &program_CUBE,
                        &mut init_encoder,
                        wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: true,
                                },
                            }],
                            depth_stencil_attachment: None,
                        },
                    ); */

                    let context2_cube = (&context_cube).set_a_Pos(&mut rpass, &positions_cube);

                    {
                        let context3_cube = context2_cube.set_u_Transform(&mut rpass, &trans_mat);
                        {
                            let context4_cube =
                                context3_cube.set_a_TexCoord(&mut rpass, &texture_coords);
                            {
                                let context5_cube =
                                    context4_cube.set_t_Color_s_Color(&mut rpass, &bind_group_t_s);

                                {
                                    context5_cube
                                        .runnable(|| graphics_run_indices(&mut rpass, &index_cube, 1));
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
