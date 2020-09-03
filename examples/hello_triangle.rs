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
    compile_buffer, default_bind_group, generate_swap_chain, setup_render_pass,
    valid_fragment_shader, valid_vertex_shader, GraphicsBindings, GraphicsShader,
    OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_vec3, is_gl_builtin, ready_to_run, update_bind_context, Bindings,
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

    const S_V: GraphicsShader = VERTEXT.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = VERTEXT.1;
    const S_F: GraphicsShader = FRAGMENT.0;

    let (program, template_bindings, template_out_bindings, _) =
        compile_valid_graphics_program!(window, S_V, S_F);

    let positions = vec![[0.0, 0.7, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]];
    let brightness = vec![0.5, 0.5, 0.9];

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = generate_swap_chain(&program, &window);

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                let mut init_encoder = program
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let mut frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");
                let mut rpass = setup_render_pass(&program, &mut init_encoder, &frame);
                let mut bind_group = default_bind_group(&program);

                let mut bindings: GraphicsBindings = template_bindings.clone();
                let mut out_bindings: OutGraphicsBindings = template_out_bindings.clone();
                const BIND_CONTEXT_1: [&str; 32] =
                    update_bind_context(&STARTING_BIND_CONTEXT, "in_brightness");
                bind_fvec(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &brightness,
                    "in_brightness".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context(&BIND_CONTEXT_1, "a_position");
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
                            rpass,
                            &mut bind_group,
                            &bindings,
                            &out_bindings,
                        );
                    }
                }
                program.queue.submit(&[init_encoder.finish()]);
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
