use glsl_to_spirv::ShaderType;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;

use crate::shared::{
    check_gl_builtin_type, compile_shader, has_out_qual, is_gl_builtin, process_body,
    string_compare, GLSLTYPE, PARAMETER, QUALIFIER,
};

use crate::bind::{DefaultBinding, Indices, SamplerBinding, TextureBinding};

pub struct GraphicsProgram {
    pub pipeline: wgpu::RenderPipeline,
}

#[derive(Debug, Clone)]
pub struct GraphicsBindings {
    pub bindings: Vec<DefaultBinding>,
    pub indices: Option<Rc<wgpu::Buffer>>,
    pub index_len: Option<u32>,
    pub textures: Vec<TextureBinding>,
    pub samplers: Vec<SamplerBinding>,
}

#[derive(Debug, Clone)]
pub struct OutGraphicsBindings {
    pub bindings: Vec<DefaultBinding>,
}

fn stringify_shader(
    s: &GraphicsShader,
    b: &GraphicsBindings,
    b_out: &OutGraphicsBindings,
) -> String {
    let mut buffer = Vec::new();
    for i in &b.bindings[..] {
        if i.qual.contains(&QUALIFIER::UNIFORM) {
            buffer.push(format!(
                "layout(set = {}, binding = {}) uniform UNIFORM{}{} {{\n\t {} {};\n}};\n",
                i.group_number, i.binding_number, i.group_number, i.binding_number, i.gtype, i.name
            ));
        } else if !is_gl_builtin(&i.name) {
            buffer.push(format!(
                "layout(location={}) {} {} {};\n",
                i.binding_number,
                if i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                    "inout"
                } else if i.qual.contains(&QUALIFIER::IN) {
                    "in"
                } else if i.qual.contains(&QUALIFIER::OUT) {
                    "out"
                } else {
                    panic!("You are trying to do something with something that isn't an in or out")
                },
                i.gtype,
                i.name
            ));
        }
    }
    for i in &b.textures[..] {
        buffer.push(format!(
            "layout(set = {}, binding = {}) uniform {} {};\n",
            i.group_number, i.binding_number, i.gtype, i.name
        ));
    }
    for i in &b.samplers[..] {
        buffer.push(format!(
            "layout(set = {}, binding = {}) uniform {} {};\n",
            i.group_number, i.binding_number, i.gtype, i.name
        ));
    }
    for i in &b_out.bindings[..] {
        if i.name != "gl_Position" && !i.qual.contains(&QUALIFIER::UNIFORM) {
            buffer.push(format!(
                "layout(location={}) {} {} {};\n",
                i.binding_number,
                if i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                    "inout"
                } else if i.qual.contains(&QUALIFIER::IN) {
                    "in"
                } else if i.qual.contains(&QUALIFIER::OUT) {
                    "out"
                } else {
                    panic!("You are trying to do something with something that isn't an in or out")
                },
                i.gtype,
                i.name
            ));
        }
    }
    format!(
        //todo figure out how to use a non-1 local size
        "\n#version 450\n{}\n\n{}",
        buffer.join(""),
        process_body(s.body)
    )
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

//todo clean this up, or better yet, can this move to proc_macro?
fn create_bindings(
    vertex: &GraphicsShader,
    fragment: &GraphicsShader,
) -> (
    GraphicsBindings,
    OutGraphicsBindings,
    GraphicsBindings,
    OutGraphicsBindings,
) {
    let mut vertex_binding_struct = Vec::new();
    let mut vertex_out_binding_struct = Vec::new();
    let mut fragment_binding_struct = Vec::new();
    let mut fragment_out_binding_struct = Vec::new();
    let mut vertex_stage_binding_number = 0;
    let mut vertex_binding_number = 0;
    let mut vertex_to_fragment_binding_number = 0;
    let mut vertex_to_fragment_map = HashMap::new();
    let mut uniform_map = HashMap::new();
    let mut uniform_binding_number_map = HashMap::new();
    let mut uniform_binding_number_fragment_map = HashMap::new();
    let mut fragment_out_binding_number = 0;
    let mut group_map: HashMap<&str, u32> = HashMap::new();
    let mut group_set_number = 0;
    for i in &vertex.params[..] {
        if !check_gl_builtin_type(i.name, &i.gtype) {
            // Bindings that are kept between runs
            if i.qual.contains(&QUALIFIER::VERTEX) {
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: vertex_binding_number,
                    group_number: 0,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                vertex_binding_number += 1;
            // Bindings that are invalidated after a run
            } else if i.qual.contains(&QUALIFIER::UNIFORM) {
                let group_number = match group_map.get(i.group.unwrap()) {
                    Some(i) => *i,
                    None => {
                        let x = group_set_number;
                        group_map.insert(i.group.unwrap(), x);
                        group_set_number += 1;
                        x
                    }
                };
                let uniform_binding_number =
                    uniform_binding_number_map.entry(group_number).or_insert(0);
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: *uniform_binding_number,
                    group_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                uniform_map.insert(i.group.unwrap(), *uniform_binding_number);
                *uniform_binding_number += 1;
            } else if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: vertex_stage_binding_number,
                    group_number: 0,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                vertex_stage_binding_number += 1;
            // Bindings that are invalidated after a run
            } else if !i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                vertex_out_binding_struct.push(DefaultBinding {
                    binding_number: vertex_to_fragment_binding_number,
                    group_number: 0,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                vertex_to_fragment_map.insert(i.name, vertex_to_fragment_binding_number);
                vertex_to_fragment_binding_number += 1;
            } else {
                dbg!(&i);
                panic!("TODO We currently don't support both in and out qualifiers for vertex/fragment shaders")
            }
        }
    }

    let mut textures_struct = Vec::new();
    let mut samplers_struct = Vec::new();

    for i in &fragment.params[..] {
        //todo the binding_number's below are probably wrong
        if !check_gl_builtin_type(i.name, &i.gtype) {
            // Bindings that are kept between runs
            if i.qual.contains(&QUALIFIER::UNIFORM) {
                if i.gtype == GLSLTYPE::Sampler || i.gtype == GLSLTYPE::SamplerShadow {
                    let group_number = match group_map.get(i.group.unwrap()) {
                        Some(i) => *i,
                        None => {
                            let x = group_set_number;
                            group_map.insert(i.group.unwrap(), x);
                            group_set_number += 1;
                            x
                        }
                    };
                    let binding_number = uniform_binding_number_fragment_map
                        .entry(group_number)
                        .or_insert(0);
                    samplers_struct.push(SamplerBinding {
                        binding_number: *binding_number,
                        group_number,
                        name: i.name.to_string(),
                        data: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    *binding_number += 1;
                } else if i.gtype == GLSLTYPE::Texture2D
                    || i.gtype == GLSLTYPE::Texture2DArray
                    || i.gtype == GLSLTYPE::TextureCube
                {
                    let group_number = match group_map.get(i.group.unwrap()) {
                        Some(i) => *i,
                        None => {
                            let x = group_set_number;
                            group_map.insert(i.group.unwrap(), x);
                            group_set_number += 1;
                            x
                        }
                    };
                    let binding_number = uniform_binding_number_fragment_map
                        .entry(group_number)
                        .or_insert(0);
                    textures_struct.push(TextureBinding {
                        binding_number: *binding_number,
                        group_number,
                        name: i.name.to_string(),
                        data: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    *binding_number += 1;
                } else {
                    let group_number = match group_map.get(i.group.unwrap()) {
                        Some(i) => *i,
                        None => {
                            let x = group_set_number;
                            group_map.insert(i.group.unwrap(), x);
                            group_set_number += 1;
                            x
                        }
                    };
                    let binding_number = uniform_binding_number_fragment_map
                        .entry(group_number)
                        .or_insert(0);
                    /* let num = if uniform_map.get(i.name).is_some() {
                        *uniform_map.get(i.name).unwrap()
                    } else {
                        uniform_map
                        0
                        /* let x = uniform_binding_number;
                        uniform_map.insert(i.name, x);
                        uniform_binding_number += 1;
                        x */
                    }; */
                    fragment_binding_struct.push(DefaultBinding {
                        binding_number: *binding_number,
                        group_number,
                        name: i.name.to_string(),
                        data: None,
                        length: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    *binding_number += 1;
                }
            } else if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                fragment_binding_struct.push(DefaultBinding {
                    binding_number: *vertex_to_fragment_map
                        .get(i.name)
                        .unwrap_or_else(|| panic!("{} has not been bound", i.name)),
                    group_number: 0,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
            // Bindings that are invalidated after a run
            } else if !i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                fragment_out_binding_struct.push(DefaultBinding {
                    binding_number: fragment_out_binding_number,
                    group_number: 0,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                fragment_out_binding_number += 1;
            } else {
                //panic!("TODO We currently don't support both in and out qualifiers for vertex/fragment shaders")
            }
        }
    }

    (
        GraphicsBindings {
            bindings: vertex_binding_struct,
            indices: None,
            index_len: None,
            textures: Vec::new(),
            samplers: Vec::new(),
        },
        OutGraphicsBindings {
            bindings: vertex_out_binding_struct,
        },
        GraphicsBindings {
            bindings: fragment_binding_struct,
            indices: None,
            index_len: None,
            textures: textures_struct,
            samplers: samplers_struct,
        },
        OutGraphicsBindings {
            bindings: fragment_out_binding_struct,
        },
    )
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
    vertex: &GraphicsShader,
    fragment: &GraphicsShader,
    args: GraphicsCompileArgs,
) -> GraphicsProgram {
    // the adapter is the handler to the physical graphics unit

    let (program_bindings1, out_program_bindings1, program_bindings2, out_program_bindings2) =
        create_bindings(&vertex, &fragment);

    for i in &program_bindings1.bindings[..] {
        if i.qual.contains(&QUALIFIER::VERTEX) {
            vec_buffer[i.binding_number as usize] = wgpu::VertexAttribute {
                offset: 0,
                // This is our connection to shader.vert
                // TODO WOW I had an error because I hardcoded the format's below. That should not be a thing
                shader_location: i.binding_number,
                format: if i.gtype == GLSLTYPE::Vec3 {
                    wgpu::VertexFormat::Float3
                } else if i.gtype == GLSLTYPE::Vec2 {
                    wgpu::VertexFormat::Float2
                } else {
                    wgpu::VertexFormat::Float
                },
            };
        }
    }

    let mut vertex_binding_desc = Vec::new();

    for i in &program_bindings1.bindings[..] {
        if !i.qual.contains(&QUALIFIER::UNIFORM) && !i.qual.contains(&QUALIFIER::BUFFER) {
            vertex_binding_desc.push(wgpu::VertexBufferLayout {
                array_stride: (i.gtype.size_of()) as wgpu::BufferAddress,
                step_mode: if i.qual.contains(&QUALIFIER::VERTEX) {
                    wgpu::InputStepMode::Vertex
                } else {
                    wgpu::InputStepMode::Instance
                },
                // If you have a struct that specifies your vertex, this is a 1 to 1 mapping of that struct
                attributes: &vec_buffer
                    [((i.binding_number) as usize)..((i.binding_number + 1) as usize)],
            });
        }
    }

    let x = stringify_shader(vertex, &program_bindings1, &out_program_bindings1);

    println!("{}", x);

    // Our compiled vertex shader
    let vs_module = compile_shader(x, ShaderType::Vertex, &device);

    let y = stringify_shader(fragment, &program_bindings2, &out_program_bindings2);

    println!("{}", y);

    // Our compiled fragment shader
    let fs_module = compile_shader(y, ShaderType::Fragment, &device);

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
            module: &vs_module,
            // The name of the method in shader.vert to use
            entry_point: "main",
            buffers: &vertex_binding_desc[..],
        },
        // Notice how the fragment and rasterization parts are optional
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
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
/* todo
pub fn graphics_pipe(
    program: &GraphicsProgram,
    rpass: wgpu::RenderPass,
    bind_group: &mut wgpu::BindGroup,
    mut in_bindings: GraphicsBindings,
    mut out_bindings: &mut OutGraphicsBindings,
    result_vec: Vec<DefaultBinding>,
) {
    for i in result_vec {
        let binding = match in_bindings.bindings.iter().position(|x| x.name == i.name) {
            Some(x) => &mut in_bindings.bindings[x],
            None => {
                let x = out_bindings
                    .bindings
                    .iter()
                    .position(|x| x.name == i.name)
                    .expect("We couldn't find the binding");
                &mut out_bindings.bindings[x]
            }
        };

        /*          todo Check the types somewhere
        if !acceptable_types.contains(&binding.gtype) {
            panic!(
                "The type of the value you provided is not what was expected, {:?}",
                &binding.gtype
            );
        } */

        binding.data = Some(i.data.unwrap());
        binding.length = Some(i.length.unwrap());
    }

    graphics_run(program, rpass, bind_group, &in_bindings, out_bindings);
} */

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

#[derive(Debug)]
pub struct GraphicsShader {
    pub params: &'static [PARAMETER],
    pub body: &'static str,
}

pub const fn valid_vertex_shader(vert: &GraphicsShader) {
    let mut acc = 0;
    while acc < vert.params.len() {
        if string_compare(vert.params[acc].name, "gl_Position")
            && has_out_qual(vert.params[acc].qual)
        {
            if let GLSLTYPE::Vec4 = vert.params[acc].gtype {
                return;
            }
        }
        acc += 1;
    }
    panic!("This is not a valid vertex shader! Remember you need 'gl_Position' as an out of a vertex shader")
}

pub const fn valid_fragment_shader(frag: &GraphicsShader) {
    let mut acc = 0;
    while acc < frag.params.len() {
        if string_compare(frag.params[acc].name, "color") && has_out_qual(frag.params[acc].qual) {
            if let GLSLTYPE::Vec4 = frag.params[acc].gtype {
                return;
            }
        }
        acc += 1;
    }
    panic!("This is not a valid fragment shader! Remember you need 'color' as an out of a fragment shader")
}

#[macro_export]
macro_rules! graphics_shader {
    ($($body:tt)*) => {{
        const S : (&[pipeline::shared::PARAMETER], &'static str) = shader!($($body)*);
        (pipeline::wgpu_graphics_header::GraphicsShader{params:S.0, body:S.1})
    }};
}

#[macro_export]
macro_rules! compile_valid_graphics_program {
    ($device:tt, $context:tt, $vertex:tt, $fragment:tt, $args:expr) => {{
        let mut compile_buffer: [wgpu::VertexAttribute; 32] =
            pipeline::wgpu_graphics_header::compile_buffer();

        const _: () = pipeline::wgpu_graphics_header::valid_vertex_shader(&$vertex);
        const _: () = pipeline::wgpu_graphics_header::valid_fragment_shader(&$fragment);
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
