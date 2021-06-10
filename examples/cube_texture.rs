#![recursion_limit = "1024"]
#![feature(const_generics)]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;
use std::rc::Rc;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub use pipeline::wgpu_graphics_header::{
    generate_swap_chain, graphics_run_indices, setup_render_pass, GraphicsCompileArgs,
};

pub use wgpu_macros::generic_bindings;

use crate::pipeline::AbstractBind;
pub use pipeline::bind::{
    BindGroup1, BindGroup2, BufferData, Indices, SamplerData, TextureData, Vertex,
};

pub use pipeline::helper::{
    create_texels, generate_projection_matrix, generate_view_matrix, load_cube,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

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
            [[location(0)]] tex_coord: vec2<f32>;
            [[builtin(position)]] position: vec4<f32>;
        };

        [[block]]
        struct Locals {
            transform: mat4x4<f32>;
        };
        [[group(0), binding(0)]]
        var r_locals: Locals;

        [[stage(vertex)]]
        fn vs_main(
            [[location(0)]] position: vec3<f32>,
            [[location(1)]] tex_coord: vec2<f32>,
        ) -> VertexOutput {
            var out: VertexOutput;
            out.tex_coord = tex_coord;
            out.position = r_locals.transform * vec4<f32>(position.x,position.y,position.z,1.0);
            return out;
        }

        [[group(0), binding(1)]]
        var r_color: texture_2d<u32>;

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
            let tex = textureLoad(r_color, vec2<i32>(in.tex_coord * 256.0), 0);
            let v = f32(tex.x) / 255.0;
            return vec4<f32>(1.0 - (v * 5.0), 1.0 - (v * 15.0), 1.0 - (v * 50.0), 1.0);
        }

        [[stage(fragment)]]
        fn fs_wire() -> [[location(0)]] vec4<f32> {
            return vec4<f32>(0.0, 0.5, 0.0, 0.5);
        }
    }}

    eager_binding! {context = pipeline!()};

    /*
    struct Locals<const BINDINGTYPE: wgpu::BufferBindingType> { transform : cgmath :: Matrix4 < f32 >, }
    */

    impl<const BINDINGTYPE: wgpu::BufferBindingType> pipeline::bind::WgpuType for Locals<BINDINGTYPE> {
        fn bind(
            &self,
            device: &wgpu::Device,
            qual: Option<pipeline::shared::QUALIFIER>,
        ) -> pipeline::bind::BoundData {
            use pipeline::align::Alignment;
            pipeline::bind::BoundData::new_buffer(
                device,
                self.transform.align_bytes(),
                1 as u64,
                Self::size_of(),
                qual,
                Self::create_binding_type(),
            )
        }
        fn size_of() -> usize {
            use pipeline::align::Alignment;
            <f32>::alignment_size()
        }
        fn create_binding_type() -> wgpu::BindingType {
            wgpu::BindingType::Buffer {
                ty: BINDINGTYPE,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
            }
        }
        fn get_qualifiers() -> Option<pipeline::shared::QUALIFIER> {
            match BINDINGTYPE {
                wgpu::BufferBindingType::Uniform => Some(pipeline::shared::QUALIFIER::UNIFORM),
                wgpu::BufferBindingType::Storage { read_only: _ } => {
                    Some(pipeline::shared::QUALIFIER::BUFFER)
                }
            }
        }
    }

    let (program, _) =
        compile_valid_graphics_program!(device, context, GraphicsCompileArgs::default());

    let queue = Rc::new(queue);

    let (positions, _, index_data) = load_cube();
    let texture_coordinates = BufferData::new(vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]);

    let vertex_position = Vertex::new(&device, &BufferData::new(positions));
    let vertex_tex_coords = Vertex::new(&device, &texture_coordinates);
    let indices = Indices::new(&device, &index_data);

    let locals = Locals {
        transform: generate_projection_matrix(size.width as f32 / size.height as f32) * generate_view_matrix(),
    };

    /* let sampler = SamplerData::new(wgpu::SamplerDescriptor {
        label: Some("sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    }); */

    let tex_size = 256u32;
    let texture = TextureData::new(
        create_texels(tex_size as usize),
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        },
        wgpu::TextureViewDescriptor::default(),
        queue.clone(),
    );

    let bind_group_locals = BindGroup2::new(&device, &locals, &texture);

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
                        let context2 = context1.set_tex_coord(&mut rpass, &vertex_tex_coords);
                        {
                            let context3 = context2.set_r_locals_r_color(&mut rpass, &bind_group_locals);

                            {
                                /* let context4 =
                                context3.set_t_Color_s_Color(&mut rpass, &bind_group_t_s_map); */

                                {
                                    let _ = context3.runnable(&mut rpass, |r| {
                                        graphics_run_indices(r, &indices, 1)
                                    });
                                }
                            }
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
