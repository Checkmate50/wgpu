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
    generate_swap_chain, graphics_run, setup_render_pass, GraphicsCompileArgs,
};

use crate::pipeline::bind::WgpuType;
use crate::pipeline::AbstractBind;
pub use pipeline::bind::{BufferData, Vertex};

pub use wgpu_macros::generic_bindings;

async fn run(event_loop: EventLoop<()>, window: Window) {
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

    my_shader! {pipeline = {

        struct VertexOutput {
            [[location(0)]] posColor: vec4<f32>;
            [[location(1)]] brightness: f32;
            [[builtin(position)]] position: vec4<f32>;
        };


        [[stage(vertex)]]
        fn vs_main([[location(0)]] position: vec4<f32>, [[location(1)]] brightness: f32) -> VertexOutput {
            var out: VertexOutput;
            out.posColor = position;
            out.position = position;
            out.brightness = brightness;
            return out;
        }

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
            return in.posColor * in.brightness;
        }
    }}

    eager_binding! {context = pipeline!()};

    let (program, _) =
        compile_valid_graphics_program!(device, context, GraphicsCompileArgs::default());

    // my_shader (string) -> eager_binding(naga-> u32->u8) -> get_module(u8) -> compile_valid_graphics_program (wgpu::ShaderModule)

    let positions = vec![
        [0.0, 0.7, 0.0, 1.0],
        [0.5, -0.5, 0.0, 1.0],
        [-0.5, 0.5, 0.0, 1.0],
    ];
    let brightness = vec![0.5, 0.5, 0.9];

    let vertex_position = Vertex::new(&device, &BufferData::new(positions));

    let vertex_brightness = Vertex::new(&device, &BufferData::new(brightness));

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

                    let context1 = (&context).set_position(&mut rpass, &vertex_position);
                    {
                        let context2 = context1.set_brightness(&mut rpass, &vertex_brightness);
                        {
                            context2.runnable(&mut rpass, |r| graphics_run(r, 3, 1));
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
