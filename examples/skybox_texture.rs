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
pub use pipeline::shared::{bind_fvec, bind_vec3, new_bind_scope, ready_to_run, Bindings};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    const VERTEXT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[uniform in] mat4] proj;
        [[uniform in] mat4] view;
        [[out] vec3] v_Uv;
        [[out] vec4] gl_Position;
        [[] uvec3] gl_GlobalInvocationID;
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
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[uniform in] textureCube] t_Cubemap
        [[uniform in] sampler] s_Cubemap;
        [[in] vec3] v_View;
        [[out] vec4] color;
        {{
            void main() {
               color = texture(samplerCube(t_Cubemap, s_Cubemap), v_Uv);
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

    fn generate_uniforms(aspect_ratio: f32) -> (cgmath::Matrix4<f32>, cgmath::Matrix4<f32>) {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_correction = framework::OPENGL_TO_WGPU_MATRIX;
        (mx_correction * mx_projection, mx_correction * mx_view)
    }

    let (uniform_proj, uniform_view) = generate_uniforms(size);

    let sampler = program.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare: wgpu::CompareFunction::Undefined,
    });

    let paths: [&'static [u8]; 6] = [
        &include_bytes!("images/posx.png")[..],
        &include_bytes!("images/negx.png")[..],
        &include_bytes!("images/posy.png")[..],
        &include_bytes!("images/negy.png")[..],
        &include_bytes!("images/posz.png")[..],
        &include_bytes!("images/negz.png")[..],
    ];

    let (mut image_width, mut image_height) = (0, 0);
    let faces = paths
        .iter()
        .map(|png| {
            let png = std::io::Cursor::new(png);
            let decoder = png::Decoder::new(png);
            let (info, mut reader) = decoder.read_info().expect("can read info");
            image_width = info.width;
            image_height = info.height;
            let mut buf = vec![0; info.buffer_size()];
            reader.next_frame(&mut buf).expect("can read png frame");
            buf
        })
        .collect::<Vec<_>>();

    let texture_extent = wgpu::Extent3d {
        width: image_width,
        height: image_height,
        depth: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        array_layer_count: 6,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SKYBOX_FORMAT,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        label: None,
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
