#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use static_assertions::const_assert;

pub use pipeline::wgpu_graphics_header;
pub use pipeline::wgpu_graphics_header::{
    compile_buffer, valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run, Bindings,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
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
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[in] vec3] posColor;
        [[in] float] brightness;
        [[out] vec4] color;
        {{
            void main() {
                color = vec4(posColor * brightness, 1.0);
            }
        }}
    };

    const S_v: GraphicsShader = VERTEXT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_f: GraphicsShader = FRAGMENT.0;

    let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    static_assertions::const_assert!(valid_vertex_shader(&S_v));
    static_assertions::const_assert!(valid_fragment_shader(&S_f));
    let (program, mut template_bindings, mut template_out_bindings) =
        wgpu_graphics_header::graphics_compile(&mut compile_buffer, &window, &S_v, &S_f).await;

    let positions = vec![[0.0, 0.7, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]];
    let brightness = vec![0.5, 0.5, 0.9];

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

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();
                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "in_brightness");
                bind_fvec(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &brightness,
                    "in_brightness".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1, "a_position");
                    bind_vec3(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        &positions,
                        "a_position".to_string(),
                    );
                    {
                        ready_to_run(BIND_CONTEXT_2);
                        wgpu_graphics_header::graphics_run(
                            &program,
                            program.device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor { label: None },
                            ),
                            &bindings,
                            out_bindings,
                            &mut swap_chain,
                        );
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
