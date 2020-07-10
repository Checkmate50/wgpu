use wgpu::ShaderModule;

use glsl_to_spirv::ShaderType;

use std::io::Read;

use std::fmt;

use std::collections::HashMap;

use std::convert::TryInto;

use zerocopy::AsBytes as _;

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

#[derive(Debug)]
struct BINDING {
    binding_number: u32,
    name: String,
    data: Option<wgpu::Buffer>,
    size: Option<u64>,
    gtype: GLSLTYPE,
    qual: Vec<QUALIFIER>
}

#[derive(Debug)]
pub struct ProgramBindings {
    bindings: Vec<BINDING>,
}

#[derive(Debug)]
pub struct OutProgramBindings {
    bindings: Vec<BINDING>,
}

fn create_bindings(
    compute: &SHADER,
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, ProgramBindings) {
    let mut binding_struct = Vec::new();
    for i in &compute.params {
        if i.qual.contains(&QUALIFIER::BUFFER) {
            binding_struct.push(BINDING {
                binding_number: i.number,
                name: i.name.clone(),
                data: None,
                size: None,
                gtype: i.gtype.clone(),
                qual: i.qual.clone(),
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub async fn compile(compute: SHADER) -> (PROGRAM, ProgramBindings) {
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

    return (
        PROGRAM {
            device,
            queue,
            pipeline,
            bind_group_layout,
        },
        program_bindings,
    );
}

fn bind<'a>(
    program: &PROGRAM,
    bindings: &mut ProgramBindings,
    data: &'a [u8],
    size: u64,
    acceptable_types: Vec<GLSLTYPE>,
    name: String,
) {
    let index = bindings
        .bindings
        .iter()
        .position(|x| x.name == name)
        .expect("You are trying to bind to something that doesn't exist");

    let binding = &mut bindings.bindings[index];
    // Todo re-enable this
    /*     if binding.data.is_some() {
        panic!("You are trying to bind to something that has already been bound");
    } */
    if !acceptable_types.contains(&binding.gtype) {
        panic!(
            "The type of the value you provided is not what was expected, {:?}",
            &binding.gtype
        );
    }

    let buffer = program.device.create_buffer_with_data(
        data,
        wgpu::BufferUsage::MAP_READ
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_SRC,
    );

    binding.data = Some(buffer);
    binding.size = Some(size);
}

pub fn bind_vec(
    program: &PROGRAM,
    bindings: &mut ProgramBindings,
    numbers: &Vec<u32>,
    name: String,
) {
    bind(
        program,
        bindings,
        numbers.as_slice().as_bytes(),
        (numbers.len() * std::mem::size_of::<u32>()) as u64,
        vec![
            GLSLTYPE::ARRAY(Box::new(GLSLTYPE::INT)),
            GLSLTYPE::ARRAY(Box::new(GLSLTYPE::UINT)),
        ],
        name,
    )
}

pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
    cpass.dispatch(length, 1, 1);
}

pub async fn run(program: &PROGRAM, new_bindings: &mut ProgramBindings) -> Vec<u32> {
    let mut encoder = program
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut empty_vec = Vec::new();
    let mut buffer_map = HashMap::new();

    for i in new_bindings.bindings.iter() {
        buffer_map.insert(
            i.binding_number,
            i.data
                .as_ref()
                .expect(&format!("{} was not bound", &i.name)),
        );
    }
    // Todo use an out annotation to find this value
    let result_buffer = 0;
    {
        for i in 0..(buffer_map.len()) {
            let b = &new_bindings.bindings[i];
            empty_vec.push(wgpu::Binding {
                binding: b.binding_number,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_map.get(&(i as u32)).expect(&format!("I assumed all bindings would be buffers but I guess that has been invalidated")),
                    range: 0..b
                        .size
                        .expect(&format!("The size of {} was not set", &b.name)),
                },
            });
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
    program.queue.submit(&[encoder.finish()]);

    // Note that we're not calling `.await` here.
    let buffer_future = buffer_map.get(&result_buffer).unwrap().map_read(
        0,
        new_bindings.bindings[(result_buffer as usize)]
            .size
            .unwrap(),
    );

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    program.device.poll(wgpu::Maintain::Wait);


    for i in 0..new_bindings.bindings.len(){
        let binding = &mut new_bindings.bindings[i];
        if binding.qual.contains(&QUALIFIER::OUT){
            binding.data = None;
            binding.size = None;
        }
    }

    if let Ok(mapping) = buffer_future.await {
        let x: Vec<u32> = mapping
            .as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        x
    } else {
        panic!("failed to run compute on gpu!")
    }


}

// TODO
// Develop the syntax for binding a variable
//     Scoping for bind operations?
//
// Bind means bind
//      Not copying data
//      Mutation is bad!
//      Check the semantics of Buffer vs Binding (Bindgroup)
//
// In and out qualifiers should do something
//
// Create an out ProgramBinding that is consumed on run

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum GLSLTYPE {
    INT,
    UINT,
    FLOAT,
    ARRAY(Box<GLSLTYPE>),
}

impl fmt::Display for GLSLTYPE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GLSLTYPE::FLOAT => write!(f, "float"),
            GLSLTYPE::INT => write!(f, "int"),
            GLSLTYPE::UINT => write!(f, "uint"),
            GLSLTYPE::ARRAY(x) => write!(f, "{}[]", x),
        }
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

pub fn array_type(gtype: GLSLTYPE, depth: i64) -> GLSLTYPE {
    if depth <= 0 {
        gtype
    } else {
        array_type(GLSLTYPE::ARRAY(Box::new(gtype)), depth - 1)
    }
}

pub fn is_glsltype(s: &str) -> bool {
    match s {
        "uint" | "int" | "float" => true,
        _ => false,
    }
}

#[derive(Debug, PartialEq, Clone)]
#[allow(dead_code)]
pub enum QUALIFIER {
    BUFFER,
    // opengl compute shaders don't have in and out variables so these are purely to try and interface at this library level
    IN,
    OUT,
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

#[derive(Debug)]
pub struct PARAMETER {
    pub qual: Vec<QUALIFIER>,
    pub gtype: GLSLTYPE,
    pub name: String,
    pub number: u32,
}

#[derive(Debug)]
pub struct SHADER {
    pub params: Vec<PARAMETER>,
    pub body: String,
}

impl fmt::Display for SHADER {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buffer = Vec::new();
        for i in &self.params[..] {
            if i.qual.contains(&QUALIFIER::BUFFER) {
                buffer.push(format!(
                    "layout(binding = {}) buffer BINDINGS{} {{\n",
                    i.number, i.number
                ));

                buffer.push(i.gtype.to_string() + &i.name + ";\n");
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

// To help view macros
// https://lukaslueg.github.io/macro_railroad_wasm_demo/
// One of many rust guides for macros
// https://danielkeep.github.io/tlborm/book/mbe-macro-rules.html
// Learn macros by example
// https://doc.rust-lang.org/stable/rust-by-example/macros.html
#[macro_export]
macro_rules! shader {
    ( $([[$($qualifier:ident)*] $type:ident $($brack:tt)*] $param:ident;)*
      void main() { $($tt:tt)* }) => {
        {
            let mut s = Vec::new();
            let mut acc = 0;

            $(  let mut qualifiers = Vec::new();
                $(qualifiers.push(qualifying!($qualifier));)*

                let mut num_brackets = 0;
                $(let _:[u8;0] = $brack;num_brackets = num_brackets + 1)*;
                s.push(wgpu_compute_header::PARAMETER{qual:qualifiers,
                                                      gtype:wgpu_compute_header::array_type(typing!($type), num_brackets),
                                                      name:stringify!($param).to_string(),
                                                      number:acc});
                acc = acc+1;
            )*

            let mut b = String::new();
            // we need to space out type tokens from the identifiers so they don't look like one word
            $(let x = stringify!($tt); if (wgpu_compute_header::is_glsltype(x)) {b.push_str(&(x.to_string() + " "));} else {b.push_str(x);} )*

            wgpu_compute_header::SHADER{params:s, body:b}
        }
    };
}
