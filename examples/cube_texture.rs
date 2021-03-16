#![recursion_limit = "1024"]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;
use std::rc::Rc;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header::{
    generate_swap_chain, graphics_run_indices, setup_render_pass, GraphicsCompileArgs,
    GraphicsShader,
};

pub use wgpu_macros::generic_bindings;

use crate::pipeline::AbstractBind;
pub use pipeline::bind::{BindGroup2, Indices, SamplerData, TextureData, Vertex};

pub use pipeline::helper::{
    create_texels, generate_projection_matrix, generate_view_matrix, load_cube,
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
        .expect("Failed to find an appropiate adapter");

    // The device manages the connection and resources of the adapter
    // The queue is a literal queue of tasks for the gpu
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    my_shader! {vertex = {
        [[vertex in] vec3] a_Pos;
        [[vertex in] vec2] a_TexCoord;
        [group1 [uniform in] mat4] u_view;
        [group1 [uniform in] mat4] u_proj;

        [[out] vec2] v_TexCoord;
        [[out] vec4] gl_Position;
        {{
            void main() {
                v_TexCoord = a_TexCoord;
                gl_Position = u_proj * u_view * vec4(a_Pos, 1.0);
            }
        }}
    }}

    my_shader! {fragment = {
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

    const S_V: GraphicsShader = eager_graphics_shader! {vertex!()};

    const S_F: GraphicsShader = eager_graphics_shader! {fragment!()};

    eager_binding! {context = vertex!(), fragment!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, S_V, S_F, GraphicsCompileArgs::default());

    let queue = Rc::new(queue);

    let (positions, _, index_data) = load_cube();
    let texture_coordinates: Vec<[f32; 2]> = vec![
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

    let vertex_position = Vertex::new(&device, &positions);
    let vertex_tex_coords = Vertex::new(&device, &texture_coordinates);
    let indices = Indices::new(&device, &index_data);

    let view_mat = generate_view_matrix();
    let proj_mat = generate_projection_matrix(size.width as f32 / size.height as f32);
    let bind_group_view_proj = BindGroup2::new(&device, &view_mat, &proj_mat);

    let sampler = SamplerData::new(wgpu::SamplerDescriptor {
        label: Some("sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });

    let tex_size = 256u32;
    let texture = TextureData::new(
        create_texels(tex_size as usize),
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
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

    let bind_group_t_s_map = BindGroup2::new(&device, &texture, &sampler);

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

                    let context1 = (&context).set_a_Pos(&mut rpass, &vertex_position);

                    {
                        let context2 = context1.set_a_TexCoord(&mut rpass, &vertex_tex_coords);
                        {
                            let context3 =
                                context2.set_u_view_u_proj(&mut rpass, &bind_group_view_proj);

                            {
                                let context4 =
                                    context3.set_t_Color_s_Color(&mut rpass, &bind_group_t_s_map);

                                {
                                    let _ = context4
                                        .runnable(|| graphics_run_indices(rpass, &indices, 1));
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
