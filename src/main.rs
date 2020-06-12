use wgpu::ShaderModule;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use glsl_to_spirv::ShaderType; // TODO upgrade to shaderc

use std::fs::File;
use std::io::Read;
use std::path::Path;

// Read in a given file that should be a certain shader type and create a shader module out of it
fn get_shader(file_name: &str, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
    assert!((Path::new(file_name)).exists());
    let mut file = File::open(file_name).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    // Convert our shader(in GLSL) to SPIR-V format
    // https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
    let mut vert_file = glsl_to_spirv::compile(&contents, shader)
        .unwrap_or_else(|_| panic!("{}: {}", "You gave a bad shader source", contents));
    let mut vs = Vec::new();
    vert_file
        .read_to_end(&mut vs)
        .expect("Somehow reading the file got interrupted");
    // Take the shader, ...,  and return
    device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap())
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    // Create a surface to draw images on
    let surface = wgpu::Surface::create(&window);

    // the adapter is the handler to the physical graphics unit
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

    // Our compiled vertex shader
    let vs_module = get_shader("src/shader.vert", ShaderType::Vertex, &device);

    // Our compiled fragment shader
    let fs_module = get_shader("src/shader.frag", ShaderType::Fragment, &device);

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    struct Vertex {
        position: [f32; 3],
        brightness: f32
    }

    const VERTICES: &[Vertex] = &[
        Vertex { position: [0.0, 0.7, 1.0], brightness: 0.0 },
        Vertex { position: [-0.5, 0.5, 1.0], brightness: 0.5 },
        Vertex { position: [0.5, -0.5, 0.0], brightness: 0.9 },
    ];
    unsafe impl bytemuck::Pod for Vertex {}
    unsafe impl bytemuck::Zeroable for Vertex {}

    // This isn't quite good enough to replace the struct
    /* let vertex_positions = &[[0.0, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]; */

    let vertex_buffer_desc =
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            // If you have a struct that specifies your vertex, this is a 1 to 1 mapping of that struct
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    // This is our connection to shader.vert
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    // This is our connection to shader.vert
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float,
                }
            ]
        };

    // Create a layout for our bindings
    // If we had textures we would use this to lay them out
/*     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
        bindings: &[],
        label: None,
    }); */

    // Bind no values to none of the bindings.
    // Use for something like textures
/*     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[],
        label: None,
    }); */

    // Set up the bindings for the pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        // If we supply a binding layout that is empty then everything crashes since the shaders use a layout
        // This layout is specifically used to specify textures often for the fragment stage
        bind_group_layouts: &[],
    });

    // The part where we actually bring it all together
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        // Set each of the stages we want to do specifying which function to use to start
        // There is an implicit numbering for each of the stages. This number is used when specifying which stage you are creating bindings for
        // vertex => 1
        // fragment => 2
        // rasterization => 3
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            // The name of the method in shader.vert to use
            entry_point: "main",
        },
        // Notice how the fragment and rasterization parts are optional
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            // The name of the method in shader.frag to use
            entry_point: "main",
        }),
        // Lays out how to process our primitives(See primitive_topology)
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            // Counter Clockwise facing(Basically back-facing)
            front_face: wgpu::FrontFace::Ccw,
            // Specify that we don't want to toss any of our primitives(triangles) based on which way they face. Useful for getting rid of shapes that aren't shown to the viewer
            // Alternatives include Front and Back culling
            // We are currently back-facing so CullMode::Front does nothing and Back gets rid of the triangle
            cull_mode: wgpu::CullMode::Back,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        // Use Triangles
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            // Specify the size of the color data in the buffer
            // Bgra8UnormSrgb is specifically used since it is guaranteed to work on basically all browsers (32bit)
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            // Here is where you can do some fancy stuff for transitioning colors/brightness between frames. Replace defaults to taking all of the current frame and none of the next frame.
            // This can be changed by specifying the modifier for either of the values from src/dest frames or changing the operation used to combine them(instead of addition maybe Max/Min)
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            // We can adjust the mask to only include certain colors if we want to
            write_mask: wgpu::ColorWrite::ALL,
        }],
        // We can add an optional stencil descriptor which allows for effects that you would see in Microsoft Powerpoint like fading/swiping to the next slide
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            // Specifies the size of our data in our non-existent buffers
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[vertex_buffer_desc],
        },
        // Number of samples to use per pixel(Use more than one for some fancy multisampling)
        sample_count: 1,
        // Use all available samples(This is a bitmask)
        sample_mask: !0,
        // Create a mask using the alpha values for each pixel and combine it with the sample mask to limit what samples are used
        alpha_to_coverage_enabled: false,
    });

    // For drawing to window
    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        // Only update during the "vertical blanking interval"
        // As opposed to Immediate where it is possible to see visual tearing(where multiple frames are visible at once)
        present_mode: wgpu::PresentMode::Mailbox,
    };

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let vertex_buffer = device.create_buffer_with_data(bytemuck::cast_slice(VERTICES), wgpu::BufferUsage::VERTEX);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
            }
            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                // So the block here is interesting. rpass here is borrowing our encoder and doing all of the drawing. But then we need to toss rpass so we can use the encoder next with encoder.finish()
                // It feels super janky
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        // color_attachments is literally where we draw the colors to
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            // The texture we are saving the colors to
                            attachment: &frame.view,
                            resolve_target: None,
                            load_op: wgpu::LoadOp::Clear,
                            store_op: wgpu::StoreOp::Store,
                            // Default color for all pixels
                            // Use Color to specify a specific rgba value
                            clear_color: wgpu::Color::WHITE,
                        }],
                        depth_stencil_attachment: None,
                    });

                    // The order must be set_pipeline -> set a bind_group if needed -> set a vertex buffer -> set an index buffer -> do draw
                    // Otherwise we crash out
                    rpass.set_pipeline(&render_pipeline);
                    /* rpass.set_bind_group(0, &bind_group, &[]); */
                    rpass.set_vertex_buffer(0, &vertex_buffer, 0, 0);
                    // Draw 3 verticies, 1 instance
                    rpass.draw(0..3, 0..1);
                }

                // Do the rendering by saying that we are done and sending it off to the gpu
                queue.submit(&[encoder.finish()]);
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
    println!("Hello, world!");
    // From examples of wgpu-rs, set up a window we can use to view our stuff
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    /* let window = winit::window::Window::new(&event_loop).unwrap(); */

    // Why do we need to be async? Because of event_loop?
    futures::executor::block_on(run(event_loop, window));
}
