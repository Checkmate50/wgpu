use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    brightness: f32,
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    let program =
        wgpu_graphics_header::graphics_compile(&window, "examples/shader.vert", "examples/shader.frag").await;

    // The data we will be using
    const VERTICES: &[Vertex] = &[
        Vertex {
            position: [0.0, 0.7, 0.0],
            brightness: 0.5,
        },
        Vertex {
            position: [-0.5, 0.5, 1.0],
            brightness: 0.5,
        },
        Vertex {
            position: [0.5, -0.5, 0.0],
            brightness: 0.9,
        },
    ];

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
    let mut swap_chain = program.device.create_swap_chain(&program.surface, &sc_desc);

    let vertex_buffer =
        program.device.create_buffer_with_data(bytemuck::cast_slice(VERTICES), wgpu::BufferUsage::VERTEX);

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
                swap_chain = program.device.create_swap_chain(&program.surface, &sc_desc);
            }
            Event::RedrawRequested(_) => {
                // So the block here is interesting. rpass here is borrowing our encoder and doing all of the drawing. But then we need to toss rpass so we can use the encoder next with encoder.finish()
                // It feels super janky
                {
                    //todo typechecker and scoping
                    /*                     with program:
                    bind ...
                        run() */
                    /* rpass.set_bind_group(0, &bind_group, &[]); */
                    /* wgpu_graphics_header::bind(&mut rpass, &vertex_buffer); */
                    // Draw 3 verticies, 1 instance
                    /*                    fn render_draw(rpass: &mut wgpu::RenderPass){
                        wgpu_header::draw(&mut rpass, 0..3, 0..1);
                    } */
                    wgpu_graphics_header::graphics_run(&program, &vertex_buffer, &mut swap_chain);
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
