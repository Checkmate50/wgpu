#![feature(never_type)]

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

use pipeline::wgpu_graphics_header::{
    generate_swap_chain, graphics_run, setup_render_pass, GraphicsCompileArgs, GraphicsShader, GraphicsProgram,
};

use crate::pipeline::AbstractBind;
use pipeline::bind::Vertex;
use wgpu::*;
use std::error::Error;

struct State {
    queue: Queue,
    device: Device,
    swap_chain: SwapChain,
    program: GraphicsProgram,
    vertex_position: Vertex<Vec<[f32; 3]>>,
    vertex_brightness: Vertex<Vec<f32>>,
    context: WhatDoIPutHere,
}

impl State {
    async fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let instance = Instance::new(BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            }).await.ok_or("Couldn't create adapter")?;

        // The device manages the connection and resources of the adapter
        // The queue is a literal queue of tasks for the gpu
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor::default(),
                None,
            )
            .await
            .expect("Failed to create device");

        my_shader! {vertex = {
            [[vertex in] vec3] a_position;
            [[vertex in] vec1] in_brightness;
            [[out] vec3] posColor;
            [[out] vec1] brightness;
            [[out] vec4] gl_Position;
            {{
                void main() {
                    posColor = a_position;
                    brightness = in_brightness;
                    gl_Position = vec4(a_position, 1.0);
                }
            }}
        }}

        my_shader! {fragment = {
            [[in] vec3] posColor;
            [[in] vec1] brightness;
            [[out] vec4] color;
            {{
                void main() {
                    color = vec4(posColor * brightness, 1.0);
                }
            }}
        }}

        const S_V: GraphicsShader = eager_graphics_shader! {vertex!()};

        const S_F: GraphicsShader = eager_graphics_shader! {fragment!()};

        eager_binding! {context = vertex!(), fragment!()};

        let (program, _) =
            compile_valid_graphics_program!(device, context, S_V, S_F, GraphicsCompileArgs::default());

        let positions = vec![[0.0, 0.7, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]];
        let brightness = vec![0.5, 0.5, 0.9];

        let vertex_position = Vertex::new(&device, &positions);
        let vertex_brightness = Vertex::new(&device, &brightness);

        // A "chain" of buffers that we render on to the display
        let swap_chain = generate_swap_chain(&surface, &window, &device);

        Ok(Self {
            device,
            queue,
            swap_chain,
            program,
            vertex_position,
            vertex_brightness,
            context,
        })
    }

    fn run(&self) -> Result<(), Box<dyn Error>> {
        let mut init_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let frame = self.swap_chain.get_current_frame()?.output;

        {
            let mut rpass = setup_render_pass(
                &self.program,
                &mut init_encoder,
                RenderPassDescriptor {
                    label: None,
                    color_attachments: &[RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::TRANSPARENT),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                },
            );

            let context1 = (&self.context).set_a_position(&mut rpass, &self.vertex_position);
            {
                let context2 = context1.set_in_brightness(&mut rpass, &self.vertex_brightness);
                {
                    context2.runnable(|| graphics_run(rpass, 3, 1));
                }
            }
        }
        self.queue.submit(Some(init_encoder.finish()));

        Ok(())
    }
}

fn main() -> Result<!, Box<dyn Error>>{
    // From examples of wgpu-rs, set up a window we can use to view our stuff
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    // Why do we need to be async? Because adapter creation is async
    let state = futures::executor::block_on(State::new(window))?;
    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // When the window closes we are done. Change the status
            Event::WindowEvent { event: WindowEvent::CloseRequested, ..  } => *control_flow = ControlFlow::Exit,
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
               match state.run() {
                   Ok(()) => (),
                   Err(e) => panic!(e.to_string()),
               }
            },
            // Ignore any other types of events
            _ => {}
        }
    });
}
