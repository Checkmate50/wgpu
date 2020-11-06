use glsl_to_spirv::ShaderType;
use zerocopy::AsBytes as _;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;

use crate::shared::{
    check_gl_builtin_type, compile_shader, has_in_qual, has_out_qual, has_uniform_qual,
    process_body, string_compare, GLSLTYPE, PARAMETER, QUALIFIER,
};

use crate::bind::{
    new_bindings, Bindings, DefaultBinding, OutProgramBindings, ProgramBindings, SamplerBinding,
    TextureBinding,
};

use crate::context::BindingContext;

pub struct GraphicsProgram {
    bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

#[derive(Debug, Clone)]
pub struct GraphicsBindings {
    pub bindings: Vec<DefaultBinding>,
    pub indicies: Option<Rc<wgpu::Buffer>>,
    pub index_len: Option<u32>,
    pub textures: Vec<TextureBinding>,
    pub samplers: Vec<SamplerBinding>,
}

impl ProgramBindings for GraphicsBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding> {
        &mut self.bindings
    }
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding {
        &mut self.bindings[index]
    }
    fn get_samplers(&mut self) -> Option<&mut Vec<SamplerBinding>> {
        Some(&mut self.samplers)
    }
    fn get_textures(&mut self) -> Option<&mut Vec<TextureBinding>> {
        Some(&mut self.textures)
    }
}

#[derive(Debug, Clone)]
pub struct OutGraphicsBindings {
    pub bindings: Vec<DefaultBinding>,
}

impl OutProgramBindings for OutGraphicsBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding> {
        &mut self.bindings
    }
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding {
        &mut self.bindings[index]
    }
}

impl Bindings for GraphicsBindings {
    fn new(&self) -> GraphicsBindings {
        GraphicsBindings {
            bindings: new_bindings(&self.bindings),
            indicies: None,
            index_len: None,
            textures: new_textures(&self.textures),
            samplers: new_samplers(&self.samplers),
        }
    }
}

impl Bindings for OutGraphicsBindings {
    fn new(&self) -> OutGraphicsBindings {
        OutGraphicsBindings {
            bindings: new_bindings(&self.bindings),
        }
    }
}

fn new_textures(bindings: &Vec<TextureBinding>) -> Vec<TextureBinding> {
    let mut new = Vec::new();

    for i in bindings.iter() {
        new.push(TextureBinding {
            name: i.name.to_string(),
            binding_number: i.binding_number,
            qual: i.qual.clone(),
            gtype: i.gtype.clone(),
            data: None,
        })
    }
    new
}

fn new_samplers(bindings: &Vec<SamplerBinding>) -> Vec<SamplerBinding> {
    let mut new = Vec::new();

    for i in bindings.iter() {
        new.push(SamplerBinding {
            name: i.name.to_string(),
            binding_number: i.binding_number,
            qual: i.qual.clone(),
            gtype: i.gtype.clone(),
            data: None,
        })
    }
    new
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
                "layout(binding = {}) uniform UNIFORM{} {{\n\t {} {};\n}};\n",
                i.binding_number, i.binding_number, i.gtype, i.name
            ));
        } else if i.name != "gl_Position" {
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
            "layout(binding = {}) uniform {} {};\n",
            i.binding_number, i.gtype, i.name
        ));
    }
    for i in &b.samplers[..] {
        buffer.push(format!(
            "layout(binding = {}) uniform {} {};\n",
            i.binding_number, i.gtype, i.name
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
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        // Only update during the "vertical blanking interval"
        // As opposed to Immediate where it is possible to see visual tearing(where multiple frames are visible at once)
        present_mode: wgpu::PresentMode::Mailbox,
    };
    device.create_swap_chain(&surface, &sc_desc)
}

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
    let mut uniform_binding_number = 0;
    let mut uniform_map = HashMap::new();
    let mut fragment_out_binding_number = 0;
    for i in &vertex.params[..] {
        if !check_gl_builtin_type(i.name, &i.gtype) {
            // Bindings that are kept between runs
            if i.qual.contains(&QUALIFIER::VERTEX) {
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: vertex_binding_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                vertex_binding_number += 1;
            // Bindings that are invalidated after a run
            } else if i.qual.contains(&QUALIFIER::UNIFORM) {
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: uniform_binding_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                uniform_map.insert(i.name, uniform_binding_number);
                uniform_binding_number += 1;
            } else if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                vertex_binding_struct.push(DefaultBinding {
                    binding_number: vertex_stage_binding_number,
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
        if !check_gl_builtin_type(i.name, &i.gtype) {
            // Bindings that are kept between runs
            if i.qual.contains(&QUALIFIER::UNIFORM) {
                if i.gtype == GLSLTYPE::Sampler || i.gtype == GLSLTYPE::SamplerShadow {
                    samplers_struct.push(SamplerBinding {
                        binding_number: uniform_binding_number,
                        name: i.name.to_string(),
                        data: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    uniform_binding_number += 1;
                } else if i.gtype == GLSLTYPE::Texture2D
                    || i.gtype == GLSLTYPE::Texture2DArray
                    || i.gtype == GLSLTYPE::TextureCube
                {
                    textures_struct.push(TextureBinding {
                        binding_number: uniform_binding_number,
                        name: i.name.to_string(),
                        data: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    uniform_binding_number += 1;
                } else {
                    let num = if uniform_map.get(i.name).is_some() {
                        uniform_map.get(i.name).unwrap().clone()
                    } else {
                        let x = uniform_binding_number;
                        uniform_map.insert(i.name, x);
                        uniform_binding_number += 1;
                        x
                    };
                    fragment_binding_struct.push(DefaultBinding {
                        binding_number: num,
                        name: i.name.to_string(),
                        data: None,
                        length: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                }
            } else if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                fragment_binding_struct.push(DefaultBinding {
                    binding_number: vertex_to_fragment_map
                        .get(i.name)
                        .unwrap_or_else(|| panic!("{} has not been bound", i.name))
                        .clone(),
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
            indicies: None,
            index_len: None,
            textures: Vec::new(),
            samplers: Vec::new(),
        },
        OutGraphicsBindings {
            bindings: vertex_out_binding_struct,
        },
        GraphicsBindings {
            bindings: fragment_binding_struct,
            indicies: None,
            index_len: None,
            textures: textures_struct,
            samplers: samplers_struct,
        },
        OutGraphicsBindings {
            bindings: fragment_out_binding_struct,
        },
    )
}

pub enum PipelineType {
    Stencil,
    ColorWithStencil,
    Color,
}

pub async fn graphics_compile(
    vec_buffer: &mut [wgpu::VertexAttributeDescriptor; 32],
    device: &wgpu::Device,
    vertex: &GraphicsShader,
    fragment: &GraphicsShader,
    pipe_type: PipelineType,
) -> (GraphicsProgram, GraphicsBindings, OutGraphicsBindings) {
    // the adapter is the handler to the physical graphics unit

    let (mut program_bindings1, out_program_bindings1, program_bindings2, out_program_bindings2) =
        create_bindings(&vertex, &fragment);

    for i in &program_bindings1.bindings[..] {
        if i.qual.contains(&QUALIFIER::VERTEX) {
            vec_buffer[i.binding_number as usize] = wgpu::VertexAttributeDescriptor {
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
    let mut bind_entry = HashMap::new();
    let mut are_bind_enties = false;

    for i in &program_bindings1.bindings[..] {
        if i.qual.contains(&QUALIFIER::UNIFORM) {
            bind_entry.insert(
                i.binding_number,
                wgpu::BindGroupLayoutEntry {
                    binding: i.binding_number,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            );
            are_bind_enties = true;
        } else if i.qual.contains(&QUALIFIER::BUFFER) {
            bind_entry.insert(
                i.binding_number,
                wgpu::BindGroupLayoutEntry {
                    binding: i.binding_number,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                    },
                },
            );
            are_bind_enties = true;
        } else {
            vertex_binding_desc.push(wgpu::VertexBufferDescriptor {
                stride: (i.gtype.size_of()) as wgpu::BufferAddress,
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

    for i in &program_bindings2.samplers[..] {
        bind_entry.insert(
            i.binding_number,
            wgpu::BindGroupLayoutEntry {
                binding: i.binding_number,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler { comparison: false },
            },
        );
        are_bind_enties = true;
    }
    for i in &program_bindings2.textures[..] {
        bind_entry.insert(
            i.binding_number,
            wgpu::BindGroupLayoutEntry {
                binding: i.binding_number,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture {
                    multisampled: false,
                    component_type: wgpu::TextureComponentType::Float,
                    dimension: wgpu::TextureViewDimension::D2,
                },
            },
        );
        are_bind_enties = true;
    }
    for i in &program_bindings2.bindings {
        if i.qual.contains(&QUALIFIER::UNIFORM) && i.qual.contains(&QUALIFIER::IN) {
            bind_entry.insert(
                i.binding_number,
                wgpu::BindGroupLayoutEntry {
                    binding: i.binding_number,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            );
            are_bind_enties = true;
        }
    }

    let bind_entry_vec = bind_entry
        .into_iter()
        .map(|(_, v)| v)
        .collect::<Vec<wgpu::BindGroupLayoutEntry>>();

    let x = stringify_shader(vertex, &program_bindings1, &out_program_bindings1);

    // Our compiled vertex shader
    let vs_module = compile_shader(x, ShaderType::Vertex, &device);

    let y = stringify_shader(fragment, &program_bindings2, &out_program_bindings2);

    // Our compiled fragment shader
    let fs_module = compile_shader(y, ShaderType::Fragment, &device);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
        bindings: if are_bind_enties {
            &bind_entry_vec
        } else {
            &[]
        },
        label: None,
    });

    let bind_group_layout_ref = [&bind_group_layout];

    // Bind no values to none of the bindings.
    // Use for something like textures
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &bind_group_layout_ref,
    });

    // The part where we actually bring it all together
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        // Set each of the stages we want to do specifying which function to use to start
        // There is an implicit numbering for each of the stages. This number is used when specifying which stage you are creating bindings for
        // vertex => 1
        // fragment => 2
        // rasterization => 3
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            // The name of the method in shader.vert to use
            entry_point: "main",
        },
        // Notice how the fragment and rasterization parts are optional
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            // The name of the method in shader.frag to use
            entry_point: "main",
        }),
        // Lays out how to process our primitives(See primitive_topology)
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            // Counter Clockwise facing(Basically back-facing)
            front_face: wgpu::FrontFace::Ccw,
            // Specify that we don't want to toss any of our primitives(triangles) based on which way they face. Useful for getting rid of shapes that aren't shown to the viewer
            // Alternatives include Front and Back culling
            // We are currently back-facing so CullMode::Front does nothing and Back gets rid of the triangle
            cull_mode: wgpu::CullMode::Back,
            depth_bias: match pipe_type {
                PipelineType::Stencil => 2,
                PipelineType::ColorWithStencil | PipelineType::Color => 0,
            },
            /// TODO We may want to adjust the depth bias and scaling during stencilling
            depth_bias_slope_scale: match pipe_type {
                PipelineType::Stencil => 2.0,
                PipelineType::ColorWithStencil | PipelineType::Color => 0.0,
            },
            depth_bias_clamp: 0.0,
        }),
        // Use Triangles
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
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

        // We can add an optional stencil descriptor which allows for effects that you would see in Microsoft Powerpoint like fading/swiping to the next slide
        depth_stencil_state: match pipe_type {
            // The first two cases are from the shadow example for the shadow pass and forward pass.
            // The last type is for typical graphics programs with no stencil
            PipelineType::Stencil => Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            PipelineType::ColorWithStencil => Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            PipelineType::Color => None,
        },
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &vertex_binding_desc[..],
        },
        // Number of samples to use per pixel(Use more than one for some fancy multisampling)
        sample_count: 1,
        // Use all available samples(This is a bitmask)
        sample_mask: !0,
        // Create a mask using the alpha values for each pixel and combine it with the sample mask to limit what samples are used
        alpha_to_coverage_enabled: false,
    });

    // TODO This is ugly, We should be able to bind across different stages and such
    program_bindings2.bindings.into_iter().for_each(|b| {
        if b.qual.contains(&QUALIFIER::UNIFORM)
            && !program_bindings1
                .bindings
                .iter()
                .any(|b2| b2.name == b.name)
        {
            program_bindings1.bindings.push(b)
        }
    });
    program_bindings1.samplers = program_bindings2.samplers;
    program_bindings1.textures = program_bindings2.textures;
    (
        GraphicsProgram {
            pipeline: render_pipeline,
            bind_group_layout,
        },
        program_bindings1,
        out_program_bindings1,
    )
}

/* pub fn bind_texture(
    bindings: &mut GraphicsBindings,
    out_bindings: &mut OutGraphicsBindings,
    texture: wgpu::TextureView,
    name: String,
) {
    let mut binding = match bindings.textures.iter().position(|x| x.name == name) {
        Some(x) => &mut bindings.textures[x],
        None => {
            panic!("I haven't considered that you would output a texture yet")
            /* let x = out_bindings
                .getBindings()
                .iter()
                .position(|x| x.name == name)
                .expect("We couldn't find the binding");
            out_bindings.samplers[x] */
        }
    };
    binding.data = Some(Rc::new(texture));
} */

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

#[derive(Debug)]
pub struct BindingPreprocess {
    indicies: Option<Rc<wgpu::Buffer>>,
    index_len: Option<u32>,
    num_instances: u32,
    verticies: Vec<(u32, Rc<wgpu::Buffer>)>,
    num_verts: u32,
    bind_group_vec: Vec<(u32, Rc<wgpu::Buffer>, std::ops::Range<u64>)>,
    texture_vec: Vec<(u32, Rc<wgpu::TextureView>)>,
    sampler_vec: Vec<(u32, Rc<wgpu::Sampler>)>,
    //bgd: wgpu::BindGroupDescriptor,
}

impl<'a> Default for BindingPreprocess {
    fn default() -> Self {
        BindingPreprocess {
            indicies: None,
            index_len: None,
            num_instances: 0,
            verticies: Vec::new(),
            num_verts: 0,
            bind_group_vec: Vec::new(),
            texture_vec: Vec::new(),
            sampler_vec: Vec::new(),
            //bgd,
        }
    }
}
impl<'a> BindingPreprocess {
    pub fn bind(
        bindings: &GraphicsBindings,
        _out_bindings: &OutGraphicsBindings, // There are not out vars that I know of for graphics pipelines so these bindings are ignored
    ) -> BindingPreprocess {
        let bind = bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::VERTEX));

        let num_verts: u32 = if bind.is_none() {
            3
        } else {
            bind.unwrap().length.unwrap() as u32
        };

        let bind = bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::LOOP));

        let num_instances: u32 = if bind.is_none() {
            1
        } else {
            bind.unwrap().length.unwrap() as u32
        };

        let mut bind_group_vec = Vec::new();
        let mut verticies = Vec::new();
        let mut textures = Vec::new();
        let mut samplers = Vec::new();

        for b in bindings.bindings.iter() {
            if b.qual.contains(&QUALIFIER::VERTEX) && b.qual.contains(&QUALIFIER::IN) {
                verticies.push((
                    b.binding_number,
                    Rc::clone(
                        b.data
                            .as_ref()
                            .expect(&format!("The binding of {} was not set", &b.name)),
                    ),
                ))
            } else {
                bind_group_vec.push((
                    b.binding_number,
                    Rc::clone(
                        b.data
                            .as_ref()
                            .expect(&format!("The binding of {} was not set", &b.name)),
                    ),
                    0..b.length
                        .expect(&format!("The size of {} was not set", &b.name)),
                ))
            }
        }

        for i in bindings.samplers.iter() {
            samplers.push((
                i.binding_number,
                Rc::clone(
                    i.data
                        .as_ref()
                        .expect(&format!("The sampler for {} was not set", &i.name)),
                ),
            ));
        }

        for i in bindings.textures.iter() {
            textures.push((
                i.binding_number,
                Rc::clone(
                    i.data
                        .as_ref()
                        .expect(&format!("The sampler for {} was not set", &i.name)),
                ),
            ));
        }

        BindingPreprocess {
            indicies: bindings.indicies.as_ref().map(|x| Rc::clone(x)),
            index_len: bindings.index_len,
            num_instances,
            verticies,
            num_verts,
            bind_group_vec,
            texture_vec: textures,
            sampler_vec: samplers,
            //bgd,
        }
    }
}

pub fn graphics_run<'a>(
    program: &GraphicsProgram,
    device: &wgpu::Device,
    mut rpass: wgpu::RenderPass<'a>,
    bind_group: &'a mut wgpu::BindGroup,
    bind_preprocess: &'a BindingPreprocess,
) -> wgpu::RenderPass<'a> {
    /* let mut encoder = program
                        .device
    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }); */

    let bind_group_bindings = &mut bind_preprocess
        .bind_group_vec
        .iter()
        .map(|(idx, buffer, len_range)| wgpu::Binding {
            binding: *idx,
            resource: wgpu::BindingResource::Buffer {
                buffer,
                range: len_range.clone(),
            },
        })
        .collect::<Vec<wgpu::Binding>>();

    bind_group_bindings.append(
        &mut bind_preprocess
            .sampler_vec
            .iter()
            .map(|(idx, sample)| wgpu::Binding {
                binding: *idx,
                resource: wgpu::BindingResource::Sampler(sample),
            })
            .collect::<Vec<wgpu::Binding>>(),
    );

    bind_group_bindings.append(
        &mut bind_preprocess
            .texture_vec
            .iter()
            .map(|(idx, texture)| wgpu::Binding {
                binding: *idx,
                resource: wgpu::BindingResource::TextureView(texture),
            })
            .collect::<Vec<wgpu::Binding>>(),
    );

    let bgd = &wgpu::BindGroupDescriptor {
        layout: &program.bind_group_layout,
        bindings: bind_group_bindings.as_slice(),
        label: None,
    };

    *bind_group = device.create_bind_group(bgd);
    {
        // The order must be set_pipeline -> set a bind_group if needed -> set a vertex buffer -> set an index buffer -> do draw
        // Otherwise we crash out
        /* rpass.set_pipeline(&program.pipeline); */

        rpass.set_bind_group(0, bind_group, &[]);

        if bind_preprocess.indicies.is_some() {
            rpass.set_index_buffer(&bind_preprocess.indicies.as_ref().unwrap(), 0, 0);
        }

        for (i, d) in bind_preprocess.verticies.iter() {
            rpass.set_vertex_buffer(*i, &d, 0, 0);
        }

        /* for i in 0..(out_bindings.bindings.len()) {
            let b = out_bindings.bindings.get(i as usize).expect(&format!(
                "I assumed all bindings would be buffers but I guess that has been invalidated"
            ));

            if b.qual.contains(&QUALIFIER::VERTEX) && b.qual.contains(&QUALIFIER::IN) {
                rpass.set_vertex_buffer(
                    b.binding_number,
                    &b.data
                        .as_ref()
                        .unwrap_or_else(|| panic!("The binding of {} was not set", &b.name)),
                    0,
                    0,
                );
            }
        } */

        if bind_preprocess.indicies.is_some() {
            draw_indexed(
                &mut rpass,
                0..bind_preprocess.index_len.unwrap(),
                0..bind_preprocess.num_instances,
            )
        } else {
            draw(
                &mut rpass,
                0..bind_preprocess.num_verts,
                0..bind_preprocess.num_instances,
            );
        }
    }
    rpass
    // Do the rendering by saying that we are done and sending it off to the gpu
    //program.queue.submit(&[encoder.finish()]);
}

pub fn graphics_run_indicies<'a>(
    program: &'a GraphicsProgram,
    device: &wgpu::Device,
    pass: wgpu::RenderPass<'a>,
    bind_group: &'a mut wgpu::BindGroup,
    bind_preprocess: &'a mut BindingPreprocess,
    indicies: &Vec<u16>,
) -> wgpu::RenderPass<'a> {
    bind_preprocess.indicies = Some(Rc::new(
        device.create_buffer_with_data(indicies.as_slice().as_bytes(), wgpu::BufferUsage::INDEX),
    ));
    bind_preprocess.index_len = Some(indicies.len() as u32);
    graphics_run(program, device, pass, bind_group, bind_preprocess)
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

pub fn default_bind_group(device: &wgpu::Device) -> wgpu::BindGroup {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
        bindings: &[],
        label: None,
    });

    let bgd = &wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[],
        label: None,
    };

    device.create_bind_group(bgd)
}

pub fn setup_render_pass<'a>(
    program: &'a GraphicsProgram,
    encoder: &'a mut wgpu::CommandEncoder,
    frame: &'a wgpu::SwapChainOutput,
) -> wgpu::RenderPass<'a> {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        // color_attachments is literally where we draw the colors to
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            // The texture we are saving the colors to
            attachment: &frame.view,
            resolve_target: None,
            load_op: wgpu::LoadOp::Clear,
            store_op: wgpu::StoreOp::Store,
            // Default color for all pixels
            // Use Color to specify a specific rgba value
            clear_color: wgpu::Color::TRANSPARENT,
        }],
        depth_stencil_attachment: None,
    });

    rpass.set_pipeline(&program.pipeline);
    rpass
}

pub fn setup_render_pass_depth<'a>(
    program: &'a GraphicsProgram,
    encoder: &'a mut wgpu::CommandEncoder,
    texture: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        // color_attachments is literally where we draw the colors to
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
            // The texture we are saving the colors to
            attachment: &texture,
            depth_load_op: wgpu::LoadOp::Clear,
            depth_store_op: wgpu::StoreOp::Store,
            stencil_load_op: wgpu::LoadOp::Clear,
            stencil_store_op: wgpu::StoreOp::Store,
            clear_depth: 1.0,
            clear_stencil: 0,
        }),
    });

    rpass.set_pipeline(&program.pipeline);
    rpass
}

pub fn setup_render_pass_color_depth<'a>(
    program: &'a GraphicsProgram,
    encoder: &'a mut wgpu::CommandEncoder,
    frame: &'a wgpu::SwapChainOutput,
    texture: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        // color_attachments is literally where we draw the colors to
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            // The texture we are saving the colors to
            attachment: &frame.view,
            resolve_target: None,
            load_op: wgpu::LoadOp::Clear,
            store_op: wgpu::StoreOp::Store,
            // Default color for all pixels
            // Use Color to specify a specific rgba value
            /* clear_color: wgpu::Color::TRANSPARENT, */
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
            // The texture we are saving the colors to
            attachment: &texture,
            depth_load_op: wgpu::LoadOp::Clear,
            depth_store_op: wgpu::StoreOp::Store,
            stencil_load_op: wgpu::LoadOp::Clear,
            stencil_store_op: wgpu::StoreOp::Store,
            clear_depth: 1.0,
            clear_stencil: 0,
        }),
    });

    rpass.set_pipeline(&program.pipeline);
    rpass
}

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
    ($device:tt, $vertex:tt, $fragment:tt, $pipe_type:path) => {{
        let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] =
            pipeline::wgpu_graphics_header::compile_buffer();

        const _: () = pipeline::wgpu_graphics_header::valid_vertex_shader(&$vertex);
        const _: () = pipeline::wgpu_graphics_header::valid_fragment_shader(&$fragment);
        let (x, y, z) = pipeline::wgpu_graphics_header::graphics_compile(
            &mut compile_buffer,
            &$device,
            &$vertex,
            &$fragment,
            $pipe_type,
        )
        .await;
        (x, y, z, compile_buffer)
    }};
}

#[macro_export]
macro_rules! compile_valid_stencil_program {
    ($device:tt, $vertex:tt, $fragment:tt, $pipe_type:path) => {{
        let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] =
            pipeline::wgpu_graphics_header::compile_buffer();

        const _: () = pipeline::wgpu_graphics_header::valid_vertex_shader(&$vertex);
        let (x, y, z) = pipeline::wgpu_graphics_header::graphics_compile(
            &mut compile_buffer,
            &$device,
            &$vertex,
            &$fragment,
            $pipe_type,
        )
        .await;
        (x, y, z, compile_buffer)
    }};
}

// todo This doesn't construct the BindingContext correctly for outs
pub const fn graphics_starting_context(
    vertex: BindingContext,
    fragment: GraphicsShader,
) -> BindingContext {
    // Take all of the in's of vertex and add the uniform in's of fragment
    let mut graphcis_bind_context = vertex;
    let mut uniforms_to_bind = [""; 32];
    let mut uniform_acc = 0;
    let mut acc = 0;

    while acc < fragment.params.len() {
        if has_in_qual(fragment.params[acc].qual) && has_uniform_qual(fragment.params[acc].qual) {
            uniforms_to_bind[uniform_acc] = fragment.params[acc].name;
            uniform_acc += 1;
        }
        acc += 1;
    }

    acc = 0;
    let mut uniform_pointer = 0; // point at the one to be added up to uniform_acc
    while acc < 32 && uniform_pointer < uniform_acc {
        if string_compare(graphcis_bind_context.starting_context[acc], "") {
            graphcis_bind_context.starting_context[acc] = uniforms_to_bind[uniform_pointer];
            uniform_pointer += 1;
        }
        acc += 1;
    }
    graphcis_bind_context
}

// This is a crazy hack
// -- I need to be able to create VertexAttributeDescriptors in compile and save a reference to them when creating the pipeline
// -- I need to somehow coerce out a 32 array from a non-copyable struct
pub fn compile_buffer() -> [wgpu::VertexAttributeDescriptor; 32] {
    let x: Box<[wgpu::VertexAttributeDescriptor]> = vec![0; 32]
        .into_iter()
        .map(|_| wgpu::VertexAttributeDescriptor {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float,
        })
        .collect();
    let y: Box<[wgpu::VertexAttributeDescriptor; 32]> = x.try_into().unwrap();
    *y
}
