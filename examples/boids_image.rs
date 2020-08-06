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
    compile_buffer, valid_fragment_shader, valid_vertex_shader, GraphicsShader,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run, Bindings,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[loop in] vec3] dstParticlePos;
        [[loop in] vec3] dstParticleVel;
        [[vertex in] vec3] trianglePos;
        [[out] vec4] gl_Position;

        {{
            void main() {
                float angle = -atan(dstParticleVel.x, dstParticleVel.y);
                vec3 pos = vec3(trianglePos.x * cos(angle) - trianglePos.y * sin(angle),
                                trianglePos.x * sin(angle) + trianglePos.y * cos(angle), 0);
                gl_Position = vec4(pos + dstParticlePos, 1);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        // isn't checked for
        [[out] vec4] color;

        {{
            void main() {
                color = vec4(1.0);
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

    let mut srcParticlePos = vec![vec![0.5, 0.0, 0.0], vec![0.3, 0.2, 0.0]];
    let mut srcParticleVel = vec![vec![0.01, -0.02, 0.0], vec![-0.05, -0.03, 0.0]];
    let triangle: Vec<Vec<f32>> = vec![
        vec![-0.01, -0.02, 0.0],
        vec![0.01, -0.02, 0.0],
        vec![0.00, 0.02, 0.0],
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

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut bindings = template_bindings.clone();
                let mut out_bindings = template_out_bindings.clone();
                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context!(STARTING_BIND_CONTEXT, "dstParticlePos");
                bind_vec3(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &srcParticlePos,
                    "dstParticlePos".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1, "dstParticleVel");
                    bind_vec3(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        &srcParticleVel,
                        "dstParticleVel".to_string(),
                    );
                    {
                        const BIND_CONTEXT_2: [&str; 32] =
                            update_bind_context!(BIND_CONTEXT_1, "trianglePos");
                        bind_vec3(
                            &program,
                            &mut bindings,
                            &mut out_bindings,
                            &triangle,
                            "trianglePos".to_string(),
                        );
                        {
                            ready_to_run(BIND_CONTEXT_2);
                            wgpu_graphics_header::graphics_run(
                                &program,
                                &bindings,
                                out_bindings,
                                &mut swap_chain,
                            );
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
