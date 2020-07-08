use wgpu::ShaderModule;

use glsl_to_spirv::ShaderType;

use std::io::Read;

use std::fmt;

// Read in a given file that should be a certain shader type and create a shader module out of it
fn compile_shader(contents: &SHADER, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
    // Convert our shader(in GLSL) to SPIR-V format
    // https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
    print!("{}", contents.to_string());
    print!("\n\n");
    let mut vert_file =
        glsl_to_spirv::compile(&contents.to_string(), shader).unwrap_or_else(|_| {
            panic!(
                "{}: {}",
                "You gave a bad shader source",
                contents.to_string()
            )
        });
    let mut vs = Vec::new();
    vert_file
        .read_to_end(&mut vs)
        .expect("Somehow reading the file got interrupted");
    // Take the shader, ...,  and return
    device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap())
}

#[derive(Debug, Clone)]
pub struct BINDING<'a> {
    binding_number: u32,
    name: String,
    pub binding: Option<wgpu::Binding<'a>>,
    // type todo
}

#[derive(Debug, Clone)]
pub struct ProgramBindings<'a> {
    pub bindings: Vec<BINDING<'a>>,
}

fn create_bindings<'a>(
    compute: &SHADER,
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, ProgramBindings<'a>) {
    let mut binding_struct = Vec::new();
    for i in &compute.params {
        if i.qual == QUALIFIER::BUFFER {
            binding_struct.push(BINDING {
                binding_number: i.number,
                name: i.name.clone(),
                binding: None,
            });
        }
    }

    // Create a layout for our bindings
    // If we had textures we would use this to lay them out

    let mut bind_entry = Vec::new();

    for i in &binding_struct {
        bind_entry.push(wgpu::BindGroupLayoutEntry {
            binding: i.binding_number,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        });
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &bind_entry,
        label: None,
    });

    return (
        bind_group_layout,
        ProgramBindings {
            bindings: binding_struct,
        },
    );
}

pub struct PROGRAM {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub async fn compile<'a>(compute: SHADER) -> (PROGRAM, wgpu::CommandEncoder, ProgramBindings<'a>) {
    // the adapter is the handler to the physical graphics unit
    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        },
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        })
        .await;

    // Our compiled vertex shader
    let cs_module = compile_shader(&compute, ShaderType::Compute, &device);

    let (bind_group_layout, program_bindings) = create_bindings(&compute, &device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    // The part where we actually bring it all together
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    return (
        PROGRAM {
            device,
            queue,
            pipeline,
            bind_group_layout,
        },
        encoder,
        program_bindings,
    );
}

pub fn with<'a>(
    program: &'a PROGRAM,
    encoder: &'a mut wgpu::CommandEncoder,
) -> wgpu::ComputePass<'a> {
    let mut cpass = encoder.begin_compute_pass();

    // The order must be set_pipeline -> set a bind_group if needed -> set a vertex buffer -> set an index buffer -> do draw
    // Otherwise we crash out
    cpass.set_pipeline(&program.pipeline);
    return cpass;
}

pub fn bind<'a>(
    bindings: &'a ProgramBindings,
    buffer: &'a wgpu::Buffer,
    size: u64,
    name: String,
) -> ProgramBindings<'a> {
    let mut new_bindings = bindings.clone();

    let index = new_bindings
        .bindings
        .iter()
        .position(|x| x.name == name)
        .expect("You are trying to bind to something that doesn't exist");

    let binding = new_bindings.bindings.remove(index);
    if binding.binding.is_some() {
        panic!("you are trying to bind to something that has already been bound");
    }

    new_bindings.bindings.push(BINDING {
        binding_number: binding.binding_number,
        name,
        binding: Some(wgpu::Binding {
            binding: binding.binding_number,
            resource: wgpu::BindingResource::Buffer {
                buffer: buffer,
                range: 0..size,
            },
        }),
    });

    return new_bindings;
}

pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
    cpass.dispatch(length, 1, 1);
}

pub fn run(encoder: &mut wgpu::CommandEncoder, program: &PROGRAM, new_bindings: ProgramBindings) {
    // TODO order on binding_number
    let mut empty_vec = Vec::new();
    for i in new_bindings.bindings.clone() {
        empty_vec.push(i.binding.expect(&i.name));
    }

    let bgd = &wgpu::BindGroupDescriptor {
        layout: &program.bind_group_layout,
        bindings: empty_vec.as_slice(),
        label: None,
    };

    let bind_group = program.device.create_bind_group(bgd);

    let mut cpass = encoder.begin_compute_pass();
    cpass.set_pipeline(&program.pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    // TODO Use loop annotations to get this number
    compute(&mut cpass, 4);
}

// TODO
// Develop the syntax for binding a variable
//     Scoping for bind operations?
// Check that all input variables for shader are bound when we run
// Extract types and do type checking
//

#[derive(Debug, Clone, PartialEq)]
pub enum GLSLTYPE {
    INT,
    UINT,
    FLOAT,
}

impl fmt::Display for GLSLTYPE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let printable = match *self {
            GLSLTYPE::FLOAT => "float",
            GLSLTYPE::INT => "int",
            GLSLTYPE::UINT => "uint",
        };
        write!(f, "{}", printable)
    }
}

#[macro_export]
macro_rules! typing {
    (uint) => {
        wgpu_compute_header::GLSLTYPE::UINT
    };
    (int) => {
        wgpu_compute_header::GLSLTYPE::INT
    };
    (float) => {
        wgpu_compute_header::GLSLTYPE::FLOAT
    };
}

pub fn is_glsltype(s: &str) -> bool {
    match s {
        "uint" | "int" | "float" => true,
        _ => false,
    }
}

#[derive(Debug, PartialEq)]
pub enum QUALIFIER {
    BUFFER,
    // opengl compute shaders don't have in and out variables so these are purely to try and interface at this library level
    IN,
    OUT,
}

#[derive(Debug)]
pub struct PARAMETER {
    pub qual: QUALIFIER,
    pub gtype: GLSLTYPE,
    pub name: String,
    pub number: u32,
}
// todo move number over to binding

#[derive(Debug)]
pub struct SHADER {
    pub params: Vec<PARAMETER>,
    pub body: String,
}

impl fmt::Display for SHADER {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buffer = Vec::new();
        for i in &self.params[..] {
            if QUALIFIER::BUFFER == i.qual {
                buffer.push(format!("layout(binding = {}) buffer BINDINGS{} {{\n", i.number, i.number));

                buffer.push(i.gtype.to_string() + "[] " + &i.name + ";\n");
                buffer.push("};\n".to_string());
            }
        }

        write!(
            fmt,
            "#version 450\nlayout(local_size_x = 1) in;\n{}\nvoid main() {{\n{}\n}}",
            buffer.join(""),
            self.body
        )
    }
}

#[macro_export]
macro_rules! qualifying {
    (buffer) => {
        wgpu_compute_header::QUALIFIER::BUFFER
    };
    (in) => {
        wgpu_compute_header::QUALIFIER::IN
    };
    (out) => {
        wgpu_compute_header::QUALIFIER::OUT
    };
}

// To help view macros
// https://lukaslueg.github.io/macro_railroad_wasm_demo/
// One of many rust guides for macros
// https://danielkeep.github.io/tlborm/book/mbe-macro-rules.html
// Learn macros by example
// https://doc.rust-lang.org/stable/rust-by-example/macros.html
#[macro_export]
macro_rules! shader {
    ( $([$qualifier:ident $type:ident $($brack:tt)*] $param:ident;)*
      void main() { $($tt:tt)* }) => {
        {
            let mut s = Vec::new();
            let mut acc = 0;
            $(s.push(wgpu_compute_header::PARAMETER{qual:qualifying!($qualifier), gtype:typing!($type), name:stringify!($param).to_string(), number:acc}); acc = acc+1;)*

            let mut b = String::new();
            // we need to space out type tokens from the identifiers so they don't look like one word
            $(let x = stringify!($tt); if (wgpu_compute_header::is_glsltype(x)) {b.push_str(&(x.to_string() + " "));} else {b.push_str(x);} )*

            wgpu_compute_header::SHADER{params:s, body:b}
        }
    };
}
