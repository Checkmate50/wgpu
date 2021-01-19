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
    bind_sampler, bind_texture, compile_buffer, default_bind_group, generate_swap_chain,
    graphics_starting_context, setup_render_pass, valid_fragment_shader, valid_vertex_shader,
    GraphicsShader,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec3, is_gl_builtin, ready_to_run, update_bind_context, Bindings,
};

pub use pipeline::helper::{generate_projection_matrix, generate_view_matrix};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
            [[uniform in] mat4] proj;
            [[uniform in] mat4] view;
            [[out] vec3] v_Uv;
            [[out] vec4] gl_Position;
    /*         [[] int] gl_VertexID; */
            {{
                void main() {
                    vec4 pos = vec4(0.0);
                    switch(gl_VertexIndex) {
                        case 0: pos = vec4(-1.0, -1.0, 0.0, 1.0); break;
                        case 1: pos = vec4( 3.0, -1.0, 0.0, 1.0); break;
                        case 2: pos = vec4(-1.0,  3.0, 0.0, 1.0); break;
                    }
                    mat3 invModelView = transpose(mat3(view));
                    vec3 unProjected = (inverse(proj) * pos).xyz;
                    v_Uv = invModelView * unProjected;

                    gl_Position = pos;
                }
            }}
        };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[uniform in] textureCube] t_Cubemap;
        [[uniform in] sampler] s_Cubemap;
        [[in] vec3] v_Uv;
        [[out] vec4] color;
        {{
            void main() {
               color = texture(samplerCube(t_Cubemap, s_Cubemap), v_Uv);
            }
        }}
    };

    const S_V: GraphicsShader = VERTEXT.0;
    const VERTEXT_STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] =
        graphics_starting_context(VERTEXT_STARTING_BIND_CONTEXT, S_F);

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(window, S_V, S_F);

    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);

    let view_mat = generate_view_matrix();

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&program, &window);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");
                let mut init_encoder = program
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                let mut bindings = template_bindings.clone();
                let mut out_bindings = template_out_bindings.clone();

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

                let paths: [&'static [u8]; 6] = [
                    &include_bytes!("images/posx.png")[..],
                    &include_bytes!("images/negx.png")[..],
                    &include_bytes!("images/posy.png")[..],
                    &include_bytes!("images/negy.png")[..],
                    &include_bytes!("images/posz.png")[..],
                    &include_bytes!("images/negz.png")[..],
                ];

                let (mut image_width, mut image_height) = (0, 0);
                let faces = paths
                    .iter()
                    .map(|png| {
                        let png = std::io::Cursor::new(png);
                        let decoder = png::Decoder::new(png);
                        let (info, mut reader) = decoder.read_info().expect("can read info");
                        image_width = info.width;
                        image_height = info.height;
                        let mut buf = vec![0; info.buffer_size()];
                        reader.next_frame(&mut buf).expect("can read png frame");
                        buf
                    })
                    .collect::<Vec<_>>();

                let texture_extent = wgpu::Extent3d {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                };

                let texture = program.device.create_texture(&wgpu::TextureDescriptor {
                    size: texture_extent,
                    array_layer_count: 6,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                    label: None,
                });

                for (i, image) in faces.iter().enumerate() {
                    let image_buf = program
                        .device
                        .create_buffer_with_data(image, wgpu::BufferUsage::COPY_SRC);

                    init_encoder.copy_buffer_to_texture(
                        wgpu::BufferCopyView {
                            buffer: &image_buf,
                            offset: 0,
                            bytes_per_row: 4 * image_width,
                            rows_per_image: 0,
                        },
                        wgpu::TextureCopyView {
                            texture: &texture,
                            mip_level: 0,
                            array_layer: i as u32,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        texture_extent,
                    );
                }

                let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    dimension: wgpu::TextureViewDimension::Cube,
                    aspect: wgpu::TextureAspect::default(),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 6,
                });

                let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);
                let mut bind_group = default_bind_group(&program);

                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context(&STARTING_BIND_CONTEXT, "t_Cubemap");
                bind_texture(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    texture_view,
                    "t_Cubemap".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context(&BIND_CONTEXT_1, "s_Cubemap");
                    bind_sampler(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        sampler,
                        "s_Cubemap".to_string(),
                    );
                    {
                        const BIND_CONTEXT_3: [&str; 32] =
                            update_bind_context(&BIND_CONTEXT_2, "view");
                        bind_mat4(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            view_mat,
                            "view".to_string(),
                        );
                        {
                            const BIND_CONTEXT_4: [&str; 32] =
                                update_bind_context(&BIND_CONTEXT_3, "proj");
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "proj".to_string(),
                            );
                            {
                                ready_to_run(BIND_CONTEXT_4);
                                wgpu_graphics_header::graphics_run(
                                    &program,
                                    rpass,
                                    &mut bind_group,
                                    &bindings,
                                    &out_bindings,
                                );
                                program.queue.submit(&[init_encoder.finish()]);
                            }
                        }
                    }
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
    /* let window = winit::window::Window::new(&event_loop).unwrap(); */

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
