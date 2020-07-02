use wgpu::ShaderModule;

use glsl_to_spirv::ShaderType;

use std::io::Read;

use std::fmt;

// Read in a given file that should be a certain shader type and create a shader module out of it
fn compile_shader(contents: SHADER, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
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

#[derive(Debug, Clone, Copy)]
struct BINDING {
    binding_number: u32,
}

fn create_bindings(binding_struct: &[BINDING]) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut bind_entry = Vec::new();

    for i in binding_struct {
        bind_entry.push(wgpu::BindGroupLayoutEntry {
            binding: i.binding_number,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        });
    }
    return bind_entry;
}

pub struct PROGRAM {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub async fn compile(compute: SHADER) -> (PROGRAM, wgpu::CommandEncoder) {
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

    // todo vertex shader -> string + main entry point + descriptors...

    // Our compiled vertex shader
    let cs_module = compile_shader(compute, ShaderType::Compute, &device);

    // todo Set up the bindings for the pipeline
    // Basically uniforms

    let binding_struct = [BINDING { binding_number: 0 }];

    let bind_entry = create_bindings(&binding_struct);

    // Create a layout for our bindings
    // If we had textures we would use this to lay them out
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &bind_entry,
        label: None,
    });

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

pub fn bind_group<'a>(cpass: &mut wgpu::ComputePass<'a>, bind_group: &'a wgpu::BindGroup) {
    cpass.set_bind_group(0, &bind_group, &[]);
}

pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
    cpass.dispatch(length, 1, 1);
}

pub fn run(cpass: &mut wgpu::ComputePass) {
    compute(cpass, 4);
}

/* shader!{
    buffer uint[] indices;
    in int TEST1;
    out float3 TEST2;
    void main() {
        uint index = gl_GlobalInvocationID.x;
        indices[index] = indices[index]+1;
    }
} */

// TODO
// Identify list of input/output variables
//     May or may not have a strict order of parameter types
// Develop the syntax for binding a variable
//     Scoping for bind operations?
//     Type level struct for bindings. Check that the binding type is not None
// Check that all input variables for shader are bound when we run
//

/*
#version 450
layout(local_size_x = 1) in;

layout(binding = 0) buffer WORD {
    uint[] indices;
}; // this is used as both input and output for convenience

void main() {
    uint index=gl_GlobalInvocationID.x;
    indices[index]=indices[index]+1;
}
*/

#[derive(Debug, Clone, PartialEq)]
pub enum GLSL_TYPE {
    INT,
    UINT,
    FLOAT,
}

impl fmt::Display for GLSL_TYPE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let printable = match *self {
            GLSL_TYPE::INT => "int",
            GLSL_TYPE::UINT => "uint",
            GLSL_TYPE::FLOAT => "float",
        };
        write!(f, "{}", printable)
    }
}

#[derive(Debug, PartialEq)]
pub enum QUALIFIER {
    BUFFER,
    IN,
    OUT,
}

#[derive(Debug)]
pub struct PARAMETER {
    pub qual: QUALIFIER,
    pub gtype: GLSL_TYPE,
    pub name: String,
}

#[derive(Debug)]
pub struct SHADER {
    pub params: Vec<PARAMETER>, // buffer_names * type => Vec<String * Type>
    // Input_names
    // Output_names
    pub body: String,
}

impl fmt::Display for SHADER {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buffer = Vec::new();
        buffer.push("layout(binding = 0) buffer BINDINGS {\n".to_string());
        for i in &self.params[..] {
            if QUALIFIER::BUFFER == i.qual {
                buffer.push(i.gtype.to_string() + "[] " + &i.name + ";\n");
            }
        }
        buffer.push("};\n".to_string());

        write!(
            fmt,
            "#version 450\nlayout(local_size_x = 1) in;\n{}\nvoid main() {{\n{}\n}}",
            buffer.join(""),
            self.body
        )
    }
}

#[macro_export]
macro_rules! typing {
    (uint) => {
        wgpu_compute_header::GLSL_TYPE::UINT
    };
    (int) => {
        wgpu_compute_header::GLSL_TYPE::INT
    };
    (float) => {
        wgpu_compute_header::GLSL_TYPE::FLOAT
    };
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
    ( $([$qualifier:ident $btype:ident $($brack:tt)*] $bparam:ident;)*
      void main() { $($tt:tt)* }) => {
        {
            let mut s = Vec::new();
            $(s.push(wgpu_compute_header::PARAMETER{qual:qualifying!($qualifier), gtype:typing!($btype), name:stringify!($bparam).to_string()});)*

            let mut b = String::new();
            // we need to space out type tokens from the identifiers so they don't look like one word
            $(let x = stringify!($tt); if (x == "uint" || x =="int" || x== "float") {b.push_str(&(x.to_string() + " "));} else {b.push_str(x);} )*

            wgpu_compute_header::SHADER{params:s, body:b}
        }
    };
    // repeat with in and out
}
