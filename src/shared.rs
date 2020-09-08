use glsl_to_spirv::ShaderType;
use regex::Regex;
use std::fmt;
use std::io::Read;
use wgpu::ShaderModule;
use zerocopy::AsBytes as _;

// Remove spaces between tokens that should be one token
// Strip off the starting and ending { }
pub fn process_body(body: &str) -> String {
    let plus = Regex::new(r"\+(\n| )*\+").unwrap();
    //println!("{:?}", body);
    let in_progress = body.strip_prefix("{").unwrap().strip_suffix("}").unwrap();
    plus.replace_all(in_progress, "++").into_owned()
}

// Read in a given file that should be a certain shader type and create a shader module out of it
pub fn compile_shader(contents: String, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
    // Convert our shader(in GLSL) to SPIR-V format
    // https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
    /*         print!("{}", contents);
    print!("\n\n"); */
    let x = glsl_to_spirv::compile(&contents, shader);
    debug!(x);
    let mut vert_file =
        x.unwrap_or_else(|_| panic!("{}: {}", "You gave a bad shader source", contents));
    let mut vs = Vec::new();
    vert_file
        .read_to_end(&mut vs)
        .expect("Somehow reading the file got interrupted");
    // Take the shader, ...,  and return
    device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap())
}

#[derive(Debug)]
pub struct DefaultBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<wgpu::Buffer>,
    pub length: Option<u64>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

pub trait ProgramBindings {
    fn getBindings(&mut self) -> &mut Vec<DefaultBinding>;
    fn indexBinding(&mut self, index: usize) -> &mut DefaultBinding;
}

pub trait OutProgramBindings {
    fn getBindings(&mut self) -> &mut Vec<DefaultBinding>;
    fn indexBinding(&mut self, index: usize) -> &mut DefaultBinding;
}

pub trait Bindings {
    fn clone(&self) -> Self;
}

pub fn new_bindings(bindings: &Vec<DefaultBinding>) -> Vec<DefaultBinding> {
    let mut new = Vec::new();

    for i in bindings.iter() {
        new.push(DefaultBinding {
            name: i.name.to_string(),
            binding_number: i.binding_number,
            qual: i.qual.clone(),
            gtype: i.gtype.clone(),
            data: None,
            length: None,
        })
    }
    new
}

pub trait Program {
    fn get_device(&self) -> &wgpu::Device;
}

fn bind_helper<'a>(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    data: &'a [u8],
    length: u64,
    acceptable_types: Vec<GLSLTYPE>,
    name: String,
) {
    let mut binding = match bindings.getBindings().iter().position(|x| x.name == name) {
        Some(x) => bindings.indexBinding(x),
        None => {
            let x = out_bindings
                .getBindings()
                .iter()
                .position(|x| x.name == name)
                .expect(&format!("We couldn't find the binding for {}", name));
            out_bindings.indexBinding(x)
        }
    };

    if !acceptable_types.contains(&binding.gtype) {
        println!("{:?}", &binding.name);
        println!("{:?}", acceptable_types);
        panic!(
            "The type of the value you provided is not what was expected, {:?}",
            &binding.gtype
        );
    }

    let buffer = program.get_device().create_buffer_with_data(
        data,
        if binding.qual.contains(&QUALIFIER::VERTEX) {
            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST
        } else if binding.qual.contains(&QUALIFIER::UNIFORM) {
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST
        } else {
            wgpu::BufferUsage::MAP_READ
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_SRC
                | wgpu::BufferUsage::VERTEX
        },
    );

    binding.data = Some(buffer);
    binding.length = Some(length);
}

pub trait Bindable {
    fn bind(
        &self,
        program: &dyn Program,
        bindings: &mut dyn ProgramBindings,
        out_bindings: &mut dyn OutProgramBindings,
        name: String,
    );
    fn bind_consume(
        &self,
        program: &dyn Program,
        bindings: dyn ProgramBindings,
        out_bindings: dyn OutProgramBindings,
        name: String,
    ) -> (&mut dyn ProgramBindings, &mut dyn OutProgramBindings);
}

impl Bindable for Vec<u32> {
    fn bind(
        &self,
        program: &dyn Program,
        bindings: &mut dyn ProgramBindings,
        out_bindings: &mut dyn OutProgramBindings,
        name: String,
    ) {
        bind_helper(
            program,
            bindings,
            out_bindings,
            self.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::ArrayInt, GLSLTYPE::ArrayUint],
            name,
        )
    }
    fn bind_consume(
        &self,
        program: &dyn Program,
        bindings: dyn ProgramBindings,
        out_bindings: dyn OutProgramBindings,
        name: String,
    ) -> (dyn ProgramBindings, dyn OutProgramBindings) {
        self.bind(program, &mut bindings, &mut out_bindings, name);
        (bindings, out_bindings)
    }
}

impl Bindable for Vec<f32> {
    fn bind(
        &self,
        program: &dyn Program,
        bindings: &mut dyn ProgramBindings,
        out_bindings: &mut dyn OutProgramBindings,
        name: String,
    ) {
        bind_helper(
            program,
            bindings,
            out_bindings,
            self.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::Float, GLSLTYPE::ArrayFloat],
            name,
        )
    }
    fn bind_consume(
        &self,
        program: &dyn Program,
        bindings: dyn ProgramBindings,
        out_bindings: dyn OutProgramBindings,
        name: String,
    ) -> (&mut dyn ProgramBindings, &mut dyn OutProgramBindings) {
        self.bind(program, &mut bindings, &mut out_bindings, name);
        (bindings, out_bindings)
    }
}

pub fn bind_vec2(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    vecs: &Vec<[f32; 2]>,
    name: String,
) {
    let numbers: Vec<f32> = vecs
        .clone()
        .into_iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect();
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        vecs.len() as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
}

pub fn bind_fvec2(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &Vec<f32>,
    name: String,
) {
    if numbers.len() % 2 != 0 {
        panic!(
            "Your trying to bind to vec to but your not giving a vector that can be split into 2's"
        )
    }
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        (numbers.len() / 2) as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
}

pub fn bind_vec3(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    vecs: &Vec<[f32; 3]>,
    name: String,
) {
    let numbers: Vec<f32> = vecs
        .clone()
        .into_iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect();
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        vecs.len() as u64,
        vec![GLSLTYPE::Vec3, GLSLTYPE::ArrayVec3],
        name,
    )
}

pub fn bind_mat4(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    mat: cgmath::Matrix4<f32>,
    name: String,
) {
    let mat_slice: &[f32; 16] = mat.as_ref();
    bind_helper(
        program,
        bindings,
        out_bindings,
        bytemuck::cast_slice(mat_slice.as_bytes()),
        64 as u64,
        vec![GLSLTYPE::Mat4],
        name,
    )
}

pub fn bind_float(
    program: &dyn Program,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &f32,
    name: String,
) {
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_bytes(),
        1 as u64,
        vec![GLSLTYPE::Float],
        name,
    )
}

// TODO functions to get the rust size and typing

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum GLSLTYPE {
    Bool,
    Int,
    Uint,
    Float,
    Vec2,
    Uvec3,
    Vec3,
    Vec4,
    Mat4,
    ArrayInt,
    ArrayUint,
    ArrayFloat,
    ArrayVec2,
    ArrayVec3,
    ArrayVec4,
    Sampler,
    TextureCube,
    Texture2D,
}

impl fmt::Display for GLSLTYPE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GLSLTYPE::Bool => write!(f, "bool"),
            GLSLTYPE::Float => write!(f, "float"),
            GLSLTYPE::Int => write!(f, "int"),
            GLSLTYPE::Uint => write!(f, "uint"),
            GLSLTYPE::Vec2 => write!(f, "vec2"),
            GLSLTYPE::Uvec3 => write!(f, "uvec3"),
            GLSLTYPE::Vec3 => write!(f, "vec3"),
            GLSLTYPE::Vec4 => write!(f, "vec4"),
            GLSLTYPE::Mat4 => write!(f, "mat4"),
            GLSLTYPE::ArrayInt => write!(f, "int[]"),
            GLSLTYPE::ArrayUint => write!(f, "uint[]"),
            GLSLTYPE::ArrayFloat => write!(f, "float[]"),
            GLSLTYPE::ArrayVec2 => write!(f, "vec2[]"),
            GLSLTYPE::ArrayVec3 => write!(f, "vec3[]"),
            GLSLTYPE::ArrayVec4 => write!(f, "vec4[]"),
            GLSLTYPE::Sampler => write!(f, "sampler"),
            GLSLTYPE::TextureCube => write!(f, "textureCube"),
            GLSLTYPE::Texture2D => write!(f, "texture2D"),
        }
    }
}

pub fn glsl_size(x: &GLSLTYPE) -> usize {
    match x {
        GLSLTYPE::Bool => std::mem::size_of::<bool>(),
        GLSLTYPE::Float => std::mem::size_of::<f32>(),
        GLSLTYPE::Int => std::mem::size_of::<i32>(),
        GLSLTYPE::Uint => std::mem::size_of::<u32>(),
        GLSLTYPE::Vec2 => std::mem::size_of::<[f32; 2]>(),
        GLSLTYPE::Uvec3 => std::mem::size_of::<[u32; 3]>(),
        GLSLTYPE::Vec3 => std::mem::size_of::<[f32; 3]>(),
        GLSLTYPE::Vec4 => std::mem::size_of::<[f32; 4]>(),
        GLSLTYPE::Mat4 => 64,
        GLSLTYPE::ArrayInt => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::ArrayUint => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::ArrayFloat => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::ArrayVec2 => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::ArrayVec3 => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::ArrayVec4 => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::Sampler => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::TextureCube => panic!("TODO: I haven't checked the size of this yet"),
        GLSLTYPE::Texture2D => panic!("TODO: I haven't checked the size of this yet"),
    }
}

#[macro_export]
macro_rules! typing {
    (bool) => {
        pipeline::shared::GLSLTYPE::Bool
    };
    (uint) => {
        pipeline::shared::GLSLTYPE::Uint
    };
    (int) => {
        pipeline::shared::GLSLTYPE::Int
    };
    (float) => {
        pipeline::shared::GLSLTYPE::Float
    };
    (vec2) => {
        pipeline::shared::GLSLTYPE::Vec2
    };
    (uvec3) => {
        pipeline::shared::GLSLTYPE::Uvec3
    };
    (vec3) => {
        pipeline::shared::GLSLTYPE::Vec3
    };
    (vec4) => {
        pipeline::shared::GLSLTYPE::Vec4
    };
    (mat4) => {
        pipeline::shared::GLSLTYPE::Mat4
    };
    (sampler) => {
        pipeline::shared::GLSLTYPE::Sampler
    };
    (textureCube) => {
        pipeline::shared::GLSLTYPE::TextureCube
    };
    (texture2D) => {
        pipeline::shared::GLSLTYPE::Texture2D
    };
}

pub const fn array_type(gtype: GLSLTYPE, depth: i64) -> GLSLTYPE {
    if depth == 1 {
        match gtype {
            GLSLTYPE::Float => GLSLTYPE::ArrayFloat,
            GLSLTYPE::Int => GLSLTYPE::ArrayInt,
            GLSLTYPE::Uint => GLSLTYPE::ArrayUint,
            GLSLTYPE::Vec2 => GLSLTYPE::ArrayVec2,
            GLSLTYPE::Vec3 => GLSLTYPE::ArrayVec3,
            GLSLTYPE::Vec4 => GLSLTYPE::ArrayVec4,
            x =>
            /* todo panic!("yikes") I want to panic but I can't as of the current nightly re;ease so we will just return itself*/
            {
                x
            }
        }
    } else {
        gtype
    }
}

#[derive(Debug, PartialEq, Clone)]
#[allow(dead_code)]
pub enum QUALIFIER {
    BUFFER,
    UNIFORM,
    VERTEX,
    // opengl compute shaders don't have in and out variables so these are purely to try and interface at this library level
    IN,
    OUT,
    LOOP,
}

// I assume there will only be one gl builtin qualifier so find that one and the match should return true
pub fn check_gl_builtin_type(p: &str, t: &GLSLTYPE) -> bool {
    match (p, t) {
        ("gl_VertexID", GLSLTYPE::Int)
        | ("gl_InstanceID", GLSLTYPE::Int)
        | ("gl_FragCoord", GLSLTYPE::Vec4)
        | ("gl_FrontFacing", GLSLTYPE::Bool)
        | ("gl_PointCoord", GLSLTYPE::Vec2)
        | ("gl_SampleID", GLSLTYPE::Int)
        | ("gl_SamplePosition", GLSLTYPE::Vec2)
        | ("gl_NumWorkGroups", GLSLTYPE::Uvec3)
        | ("gl_WorkGroupID", GLSLTYPE::Uvec3)
        | ("gl_LocalInvocationID", GLSLTYPE::Uvec3)
        | ("gl_GlobalInvocationID", GLSLTYPE::Uvec3)
        | ("gl_LocalInvocationIndex", GLSLTYPE::Uint) => true,
        _ => false,
    }
}

pub const fn string_compare(string1: &str, string2: &str) -> bool {
    let str1 = string1.as_bytes();
    let str2 = string2.as_bytes();
    if str1.len() != str2.len() {
        return false;
    }
    let mut acc = 0;
    while acc < str1.len() {
        if str1[acc] != str2[acc] {
            return false;
        }
        acc += 1;
    }
    true
}

pub const fn is_gl_builtin(p: &str) -> bool {
    if string_compare(p, "gl_VertexID")
        || string_compare(p, "gl_InstanceID")
        || string_compare(p, "gl_FragCoord")
        || string_compare(p, "gl_FrontFacing")
        || string_compare(p, "gl_PointCoord")
        || string_compare(p, "gl_SampleID")
        || string_compare(p, "gl_SamplePosition")
        || string_compare(p, "gl_NumWorkGroups")
        || string_compare(p, "gl_WorkGroupID")
        || string_compare(p, "gl_LocalInvocationID")
        || string_compare(p, "gl_GlobalInvocationID")
        || string_compare(p, "gl_LocalInvocationIndex")
    {
        true
    } else {
        false
    }
}

#[macro_export]
macro_rules! qualifying {
    (buffer) => {
        pipeline::shared::QUALIFIER::BUFFER
    };
    (uniform) => {
        pipeline::shared::QUALIFIER::UNIFORM
    };
    (vertex) => {
        pipeline::shared::QUALIFIER::VERTEX
    };
    (in) => {
        pipeline::shared::QUALIFIER::IN
    };
    (out) => {
        pipeline::shared::QUALIFIER::OUT
    };
    (loop) => {
        pipeline::shared::QUALIFIER::LOOP
    };
}

#[macro_export]
macro_rules! munch_body {
        () => {};
        ($token:tt) => {stringify!($token)};
        ($token:tt $($rest:tt)*) =>
        {
            concat!(stringify!($token), munch_body!($($rest)*))
        };
    }

#[macro_export]
macro_rules! count_brackets {
        () => {0};
        ([] $($rest:tt)*) => {1 + count_brackets!($($rest)*)};
    }

pub const fn has_in_qual(p: &[QUALIFIER]) -> bool {
    let mut acc = 0;
    while acc < p.len() {
        match p[acc] {
            QUALIFIER::IN => {
                return true;
            }
            _ => {
                acc += 1;
            }
        }
    }
    false
}

pub const fn has_out_qual(p: &[QUALIFIER]) -> bool {
    let mut acc = 0;
    while acc < p.len() {
        match p[acc] {
            QUALIFIER::OUT => {
                return true;
            }
            _ => {
                acc += 1;
            }
        }
    }
    false
}

pub const fn has_uniform_qual(p: &[QUALIFIER]) -> bool {
    let mut acc = 0;
    while acc < p.len() {
        match p[acc] {
            QUALIFIER::UNIFORM => {
                return true;
            }
            _ => {
                acc += 1;
            }
        }
    }
    false
}

#[derive(Debug)]
pub struct PARAMETER {
    pub qual: &'static [QUALIFIER],
    pub gtype: GLSLTYPE,
    pub name: &'static str,
}

// To help view macros
// https://lukaslueg.github.io/macro_railroad_wasm_demo/
// One of many rust guides for macros
// https://danielkeep.github.io/tlborm/book/mbe-macro-rules.html
// Learn macros by example
// https://doc.rust-lang.org/stable/rust-by-example/macros.html
#[macro_export]
macro_rules! shader {
    ( $([[$($qualifier:tt)*] $type:ident $($brack:tt)*] $param:ident;)*
      {$($tt:tt)*}) =>
      {
        {
            const S : &[pipeline::shared::PARAMETER] = &[$(
                pipeline::shared::PARAMETER{qual:&[$(qualifying!($qualifier)),*],
                                                      gtype:pipeline::shared::array_type(typing!($type), count_brackets!($($brack)*)),
                                                      name:stringify!($param)}),*];


            const B: &'static str = munch_body!($($tt)*);

            let mut INBINDCONTEXT  = [""; 32];
            let mut OUTBINDCONTEXT = [""; 32];
            let mut acc = 0;
            while acc < 32 {
                if acc < S.len() {
                    if !pipeline::shared::is_gl_builtin(S[acc].name){
                        if pipeline::shared::has_in_qual(S[acc].qual) {
                            INBINDCONTEXT[acc] = S[acc].name;
                        }
                        if pipeline::shared::has_out_qual(S[acc].qual) {
                            OUTBINDCONTEXT[acc] = S[acc].name;
                        }
                    }
                }
                acc += 1;
            }
            (S, B, pipeline::context::BindingContext::new(INBINDCONTEXT, OUTBINDCONTEXT))
        }
      };
    }
