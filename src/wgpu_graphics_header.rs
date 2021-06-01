use wgpu::ShaderModule;

use std::convert::TryInto;


use crate::bind::{Indices};

pub struct GraphicsProgram {
    pub pipeline: wgpu::RenderPipeline,
}

pub fn generate_swap_chain(
    surface: &wgpu::Surface,
    window: &winit::window::Window,
    device: &wgpu::Device,
) -> wgpu::SwapChain {
    let size = window.inner_size();
    // For drawing to window
    let sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    device.create_swap_chain(&surface, &sc_desc)
}

pub struct GraphicsCompileArgs {
    pub color_target_state: Option<wgpu::ColorTargetState>,
    pub primitive_state: wgpu::PrimitiveState,
    pub depth_stencil_state: Option<wgpu::DepthStencilState>,
    pub multisample_state: wgpu::MultisampleState,
}

impl Default for GraphicsCompileArgs {
    fn default() -> Self {
        GraphicsCompileArgs {
            color_target_state: Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                alpha_blend: wgpu::BlendState::default(),
                color_blend: wgpu::BlendState::default(),
                write_mask: wgpu::ColorWrite::default(),
            }),
            primitive_state: wgpu::PrimitiveState {
                cull_mode: wgpu::CullMode::Back,
                ..Default::default()
            },
            depth_stencil_state: None,
            multisample_state: wgpu::MultisampleState::default(),
        }
    }
}

pub async fn graphics_compile(
    vec_buffer: &mut [wgpu::VertexAttribute; 32],
    device: &wgpu::Device,
    bind_group_layout: Vec<wgpu::BindGroupLayout>,
    vertex_sizes: Vec<usize>,
    module: &ShaderModule,
    args: GraphicsCompileArgs,
) -> GraphicsProgram {
    for (idx, _)  in vertex_sizes.iter().enumerate() {
        vec_buffer[idx] = wgpu::VertexAttribute {
            offset: 0,
            // This is our connection to shader.vert
            // TODO WOW I had an error because I hardcoded the format's below. That should not be a thing
            shader_location: idx as u32,
            format: wgpu::VertexFormat::Float3

            /* todo if i.gtype == GLSLTYPE::Vec3 {
                wgpu::VertexFormat::Float3
            } else if i.gtype == GLSLTYPE::Vec2 {
                wgpu::VertexFormat::Float2
            } else {
                wgpu::VertexFormat::Float
            } */,
        };
    }

    let mut vertex_binding_desc = Vec::new();

    for (idx, i) in vertex_sizes.iter().enumerate() {
        vertex_binding_desc.push(wgpu::VertexBufferLayout {
            array_stride: *i as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            // If you have a struct that specifies your vertex, this is a 1 to 1 mapping of that struct
            attributes: &vec_buffer
                [idx..idx+1],
        });
    }

    let bind_group_layout_ref: Vec<&wgpu::BindGroupLayout> =
        bind_group_layout.iter().map(|a| a).collect();

    // Bind no values to none of the bindings.
    // Use for something like textures
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bind_group_layout_ref,
        push_constant_ranges: &[],
    });

    //debug!(pipeline_layout);
    debug!(vertex_binding_desc);

    let mut color_target_state = Vec::new();

    match args.color_target_state {
        Some(cts) => color_target_state.push(cts),
        None => {}
    };

    // The part where we actually bring it all together
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        // Set each of the stages we want to do specifying which function to use to start
        // There is an implicit numbering for each of the stages. This number is used when specifying which stage you are creating bindings for
        // vertex => 1
        // fragment => 2
        // rasterization => 3
        vertex: wgpu::VertexState {
            module: &module,
            // The name of the method in shader.vert to use
            entry_point: "main",
            buffers: &vertex_binding_desc[..],
        },
        // Notice how the fragment and rasterization parts are optional
        fragment: Some(wgpu::FragmentState {
            module: &module,
            // The name of the method in shader.frag to use
            entry_point: "main",
            targets: &color_target_state,
        }),
        // Use Triangles
        primitive: args.primitive_state,

        /* wgpu::PrimitiveTopology::TriangleList,
               color_states: match pipe_type {
                   PipelineType::Stencil => &[],
                   PipelineType::ColorWithStencil | PipelineType::Color => &[wgpu::ColorStateDescriptor {
                       // Specify the size of the color data in the buffer
                       // Bgra8UnormSrgb is specifically used since it is guaranteed to work on basically all browsers (32bit)
                       format: wgpu::TextureFormat::Bgra8UnormSrgb,
                       // Here is where you can do some fancy stuff for transitioning colors/brightness between frames. Replace defaults to taking all of the current frame and none of the next frame.
                       // This can be changed by specifying the modifier for either of the values from src/dest frames or changing the operation used to combine them(instead of addition maybe Max/Min)
                       color_blend: wgpu::BlendDescriptor::REPLACE,
                       alpha_blend: wgpu::BlendDescriptor::REPLACE,
                       // We can adjust the mask to only include certain colors if we want to
                       write_mask: wgpu::ColorWrite::ALL,
                   }],
               },
        */
        // We can add an optional stencil descriptor which allows for effects that you would see in Microsoft Powerpoint like fading/swiping to the next slide
        depth_stencil: args.depth_stencil_state,

        /* match pipe_type {
            // The first two cases are from the shadow example for the shadow pass and forward pass.
            // The last type is for typical graphics programs with no stencil
            PipelineType::Stencil => Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
                clamp_depth: device.features().contains(wgpu::Features::DEPTH_CLAMPING),
            }),
            PipelineType::ColorWithStencil => Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
                clamp_depth: false,
            }),
            PipelineType::Color => None,
        }, */
        multisample: args.multisample_state,
    });

    GraphicsProgram {
        pipeline: render_pipeline,
    }
}

fn draw(
    rpass: &mut wgpu::RenderPass,
    vertices: core::ops::Range<u32>,
    instances: core::ops::Range<u32>,
) {
    rpass.draw(vertices, instances);
}

fn draw_indexed(
    rpass: &mut wgpu::RenderPass,
    indexes: core::ops::Range<u32>,
    instances: core::ops::Range<u32>,
) {
    rpass.draw_indexed(indexes, 0, instances);
}

pub fn graphics_run(
    mut rpass: wgpu::RenderPass,
    num_verts: u32,
    num_instances: u32,
) -> wgpu::RenderPass {
    {
        draw(&mut rpass, 0..num_verts, 0..num_instances);
    }
    rpass
}

pub fn graphics_run_indices<'a>(
    mut rpass: wgpu::RenderPass<'a>,
    indices: &'a Indices,
    num_instances: u32,
) -> wgpu::RenderPass<'a> {
    rpass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint16);

    draw_indexed(&mut rpass, 0..indices.len, 0..num_instances);
    rpass
}

pub fn setup_render_pass<'a, 'b>(
    program: &'a GraphicsProgram,
    encoder: &'a mut wgpu::CommandEncoder,
    desc: wgpu::RenderPassDescriptor<'a, 'b>,
) -> wgpu::RenderPass<'a> {
    let mut rpass = encoder.begin_render_pass(&desc);
    rpass.set_pipeline(&program.pipeline);
    rpass
}

/*
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
    }
*/

/*
wgpu::RenderPassDescriptor {
        label: None,
        // color_attachments is literally where we draw the colors to
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
            attachment: &texture,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    }
*/

/*
wgpu::RenderPassDescriptor {
        label: None,
        // color_attachments is literally where we draw the colors to
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &frame.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                store: true,
            },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
            attachment: &texture,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    }
*/

#[macro_export]
macro_rules! compile_valid_graphics_program {
    ($device:tt, $context:tt, $args:expr) => {{
        let mut compile_buffer: [wgpu::VertexAttribute; 32] =
            pipeline::wgpu_graphics_header::compile_buffer();

        let x = pipeline::wgpu_graphics_header::graphics_compile(
            &mut compile_buffer,
            &$device,
            $context.get_bindgroup_layout(&$device),
            $context.get_vertex_sizes(),
            &$context.get_module(&$device),
            $args,
        )
        .await;
        (x, compile_buffer)
    }};
}

#[macro_export]
macro_rules! compile_valid_stencil_program {
    ($device:tt, $context:tt, $vertex:tt, $fragment:tt, $args:expr) => {{
        let mut compile_buffer: [wgpu::VertexAttribute; 32] =
            pipeline::wgpu_graphics_header::compile_buffer();

        const _: () = pipeline::wgpu_graphics_header::valid_vertex_shader(&$vertex);
        //todo maybe some validation for a fragment stencil shader?
        //todo make sure these are running at compile time
        let x = pipeline::wgpu_graphics_header::graphics_compile(
            &mut compile_buffer,
            &$device,
            $context.get_layout(&$device),
            &$vertex,
            &$fragment,
            $args,
        )
        .await;
        (x, compile_buffer)
    }};
}

// This is a crazy hack
// -- I need to be able to create VertexAttributeDescriptors in compile and save a reference to them when creating the pipeline
// -- I need to somehow coerce out a 32 array from a non-copyable struct
pub fn compile_buffer() -> [wgpu::VertexAttribute; 32] {
    let x: Box<[wgpu::VertexAttribute]> = vec![0; 32]
        .into_iter()
        .map(|_| wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float,
        })
        .collect();
    let y: Box<[wgpu::VertexAttribute; 32]> = x.try_into().unwrap();
    *y
}
