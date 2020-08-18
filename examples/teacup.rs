#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

pub use static_assertions::const_assert;

pub use pipeline::wgpu_graphics_header;
pub use pipeline::wgpu_graphics_header::{
    bind_vertex, compile_buffer, valid_fragment_shader, valid_vertex_shader, GraphicsBindings,
    GraphicsShader, OutGraphicsBindings,
};

pub use pipeline::shared;
pub use pipeline::shared::{
    bind_fvec, bind_mat4, bind_vec3, is_gl_builtin, new_bind_scope, ready_to_run, Bindings,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[vertex in] vec3] a_position;
        [[vertex in] vec3] a_normal;
        [[uniform in] vec3] Ambient;
        [[uniform in] vec3] LightDirection;
        [[uniform in] mat4] u_view;
        [[uniform in] mat4] u_proj;
        [[] int] gl_VertexID;

        [[out] vec3] fragmentNormal;
        [[out] vec4] gl_Position;
        {{
            void main() {

                vec4 worldNormal = vec4(a_normal, 0.0) * inverse(u_view);

                fragmentNormal = worldNormal.xyz;

                gl_Position = u_proj * u_view * vec4(0.7 * a_position, 1.0);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[in] vec3] fragmentNormal;
        [[uniform in] vec3] Ambient;
        [[uniform in] vec3] LightDirection;
        [[out] vec4] color;
        {{
            void main() {
                vec3 fragColor = vec3(1.0, 0.0, 0.0);
                color = vec4(Ambient + fragColor * max(dot(normalize(fragmentNormal), normalize(LightDirection)), 0.0), 1.0);
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

    let input = BufReader::new(File::open("examples/models/teapot.obj").unwrap());
    let mut dome: Obj = load_obj(input).unwrap();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = dome.indices;
    for i in &dome.vertices {
        positions.push(i.position);
        normals.push(i.normal);
    }

    let light_direction = vec![[20.0, 0.0, 10.0]];

    let light_ambient = vec![[0.1, 0.0, 0.0]];

    fn generate_view(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        mx_view
    }

    fn generate_projection(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_correction = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
        );

        mx_correction * mx_projection
    }

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

    let view_mat = generate_view(sc_desc.width as f32 / sc_desc.height as f32);

    let proj_mat = generate_projection(sc_desc.width as f32 / sc_desc.height as f32);

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
                    update_bind_context!(STARTING_BIND_CONTEXT, "a_position");
                bind_vertex(
                    &program,
                    &mut bindings,
                    &mut out_bindings,
                    &positions,
                    &indices,
                    "a_position".to_string(),
                );
                {
                    const BIND_CONTEXT_2: [&str; 32] =
                        update_bind_context!(BIND_CONTEXT_1, "u_view");
                    bind_mat4(
                        &program,
                        &mut bindings,
                        &mut out_bindings,
                        view_mat,
                        "u_view".to_string(),
                    );
                    {
                        {
                            const BIND_CONTEXT_3: [&str; 32] =
                                update_bind_context!(BIND_CONTEXT_2, "u_proj");
                            bind_mat4(
                                &program,
                                &mut bindings,
                                &mut out_bindings,
                                proj_mat,
                                "u_proj".to_string(),
                            );
                            {
                                const BIND_CONTEXT_4: [&str; 32] =
                                    update_bind_context!(BIND_CONTEXT_3, "Ambient");
                                bind_vec3(
                                    &program,
                                    &mut bindings,
                                    &mut out_bindings,
                                    &light_ambient,
                                    "Ambient".to_string(),
                                );
                                {
                                    const BIND_CONTEXT_5: [&str; 32] =
                                        update_bind_context!(BIND_CONTEXT_4, "LightDirection");
                                    bind_vec3(
                                        &program,
                                        &mut bindings,
                                        &mut out_bindings,
                                        &light_direction,
                                        "LightDirection".to_string(),
                                    );
                                    {
                                        const BIND_CONTEXT_6: [&str; 32] =
                                            update_bind_context!(BIND_CONTEXT_5, "a_normal");
                                        bind_vec3(
                                            &program,
                                            &mut bindings,
                                            &mut out_bindings,
                                            &normals,
                                            "a_normal".to_string(),
                                        );
                                        {
                                            ready_to_run(BIND_CONTEXT_6);
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
                            }
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
