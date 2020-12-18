#![recursion_limit = "512"]
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
    compile_buffer, default_bind_group, generate_swap_chain, graphics_run, setup_render_pass,
    valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings, PipelineType,
};

pub use pipeline::bind::Vertex;

pub use wgpu_macros::{generic_bindings, init};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    init!();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
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
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    my_shader! {vertex = {
        [[vertex in] vec3] a_position;
        [[vertex in] float] in_brightness;
        [[out] vec3] posColor;
        [[out] float] brightness;
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
        [[in] float] brightness;
        [[out] vec4] color;
        {{
            void main() {
                color = vec4(posColor * brightness, 1.0);
            }
        }}
    }}

    const VERTEXT: GraphicsShader = eager_graphics_shader! {vertex!()};

    const FRAGMENT: GraphicsShader = eager_graphics_shader! {fragment!()};

    //generic_bindings! {context = a_position, in_brightness; color, gl_Position}
    //eager! { lazy! { generic_bindings! { context = eager!{ vertex!(), fragment!()}}}};
    eager_binding! {context = vertex!(), fragment!()};

    const S_V: GraphicsShader = VERTEXT;
    const S_F: GraphicsShader = FRAGMENT;

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(device, S_V, S_F, PipelineType::Color);

    let positions = vec![[0.0, 0.7, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]];
    let brightness = vec![0.5, 0.5, 0.9];

    let vertex_position = Vertex::new(&device, &positions);
    let vertex_brightness = Vertex::new(&device, &brightness);

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&surface, &window, &device);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut init_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let mut frame = swap_chain
                    .get_current_frame()
                    .expect("Timeout when acquiring next swap chain texture")
                    .output;

                {
                    /* let context1 = (&context).bind_in_brightness(
                        &brightness,
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                    ); */
                    let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);
                    rpass.set_vertex_buffer(0, vertex_position.get_buffer().slice(..));
                    rpass.set_vertex_buffer(1, vertex_brightness.get_buffer().slice(..));
                    rpass = graphics_run(&device, rpass, 3, 1)

                    /* {
                                           let context2 = context1.bind_a_position(
                                               &positions,
                                               &program,
                                               &mut bindings,
                                               &mut out_bindings,
                                           );
                                           {
                                               context2.runable(|| {
                                                   graphics_run(
                                                       &program,
                                                       rpass,
                                                       &mut bind_group,
                                                       &bindings,
                                                       &out_bindings,
                                                   )
                                               });
                                           }
                                       }
                    */
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
