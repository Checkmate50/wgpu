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
    generate_swap_chain, graphics_run, setup_render_pass, GraphicsCompileArgs, GraphicsShader,
};

use crate::pipeline::AbstractBind;
pub use pipeline::bind::{BindGroup2, BufferData, SamplerData, TextureData};

pub use pipeline::helper::{generate_projection_matrix, generate_view_matrix};

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
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    my_shader! { vertex = {
        [group1 [uniform in] mat4] view;
        [group1 [uniform in] mat4] proj;
        [[out] vec3] v_Uv;
        [[out] vec4] gl_Position;
        [[] int] gl_VertexID;
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
    }}

    my_shader! { fragment = {
        [group2 [uniform in] textureCube] t_Cubemap;
        [group2 [uniform in] sampler] s_Cubemap;
        [[in] vec3] v_Uv;
        [[out] vec4] color;
        {{
            void main() {
               color = texture(samplerCube(t_Cubemap, s_Cubemap), v_Uv);
            }
        }}
    }}

    const S_V: GraphicsShader = eager_graphics_shader! {vertex!()};
    const S_F: GraphicsShader = eager_graphics_shader! {fragment!()};

    eager_binding! {context = vertex!(), fragment!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, S_V, S_F, GraphicsCompileArgs::default());

    let proj_mat = BufferData::new(generate_projection_matrix(
        size.width as f32 / size.height as f32,
    ));

    let view_mat = BufferData::new(generate_view_matrix());

    let bind_group_view_proj = BindGroup2::new(&device, &view_mat, &proj_mat);

    let sampler_desc = wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    };

    /* let texture = device.create_texture(&); */

    let paths: [&'static [u8]; 6] = [
        &include_bytes!("images/posx.png")[..],
        &include_bytes!("images/negx.png")[..],
        &include_bytes!("images/posy.png")[..],
        &include_bytes!("images/negy.png")[..],
        &include_bytes!("images/posz.png")[..],
        &include_bytes!("images/negz.png")[..],
    ];

    const IMAGE_SIZE: u32 = 512;

    let faces = paths
        .iter()
        .map(|png| {
            let png = std::io::Cursor::new(png);
            let decoder = png::Decoder::new(png);
            let (info, mut reader) = decoder.read_info().expect("can read info");
            let mut buf = vec![0; info.buffer_size()];
            reader.next_frame(&mut buf).expect("can read png frame");
            buf
        })
        .collect::<Vec<_>>();

    /* for (i, image) in faces.iter().enumerate() {
           queue.write_texture(
               wgpu::TextureCopyView {
                   texture: &texture,
                   mip_level: 0,
                   origin: wgpu::Origin3d {
                       x: 0,
                       y: 0,
                       z: i as u32,
                   },
               },
               &image,
               wgpu::TextureDataLayout {
                   offset: 0,
                   bytes_per_row: 4 * IMAGE_SIZE,
                   rows_per_image: 0,
               },
               wgpu::Extent3d {
                   width: IMAGE_SIZE,
                   height: IMAGE_SIZE,
                   depth: 1,
               },
           );
       }
    */
    /* let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..wgpu::TextureViewDescriptor::default()
    }); */

    let queue = Rc::new(queue);

    let tex = TextureData::new(
        faces.into_iter().flatten().collect(),
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: IMAGE_SIZE,
                height: IMAGE_SIZE,
                depth: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        },
        wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        },
        queue.clone(),
    );

    let sample = SamplerData::new(sampler_desc);

    let bind_group_t_s_cubemap = BindGroup2::new(&device, &tex, &sample);

    // A "chain" of buffers that we render on to the display
    let swap_chain = generate_swap_chain(&surface, &window, &device);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_current_frame()
                    .expect("Timeout when acquiring next swap chain texture")
                    .output;
                let mut init_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

                    let context1 =
                        (&context).set_t_Cubemap_s_Cubemap(&mut rpass, &bind_group_t_s_cubemap);
                    {
                        let context2 = (&context1).set_view_proj(&mut rpass, &bind_group_view_proj);
                        {
                            context2.runnable(|| graphics_run(&mut rpass, 3, 1));
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
