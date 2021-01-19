use glsl_to_spirv::ShaderType;
use regex::Regex;
use std::fmt;
use std::io::Read;
use wgpu::ShaderModule;

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
    //debug!(x);
    let mut vert_file =
        x.unwrap_or_else(|_| panic!("{}: {}", "You gave a bad shader source", contents));
    let mut vs = Vec::new();
    vert_file
        .read_to_end(&mut vs)
        .expect("Somehow reading the file got interrupted");
    // Take the shader, ...,  and return
    device.create_shader_module(wgpu::util::make_spirv(&vs[..]))
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum GLSLTYPE {
    Bool,
    Int,
    Uint,
    Float,
    Vec1,
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
    SamplerShadow,
    TextureCube,
    Texture2D,
    Texture2DArray,
}

impl GLSLTYPE {
    pub fn size_of(&self) -> usize {
        match self {
            GLSLTYPE::Bool => std::mem::size_of::<bool>(),
            GLSLTYPE::Float => std::mem::size_of::<f32>(),
            GLSLTYPE::Int => std::mem::size_of::<i32>(),
            GLSLTYPE::Uint => std::mem::size_of::<u32>(),
            GLSLTYPE::Vec1 => std::mem::size_of::<f32>(),
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
            GLSLTYPE::SamplerShadow => panic!("TODO: I haven't checked the size of this yet"),
            GLSLTYPE::TextureCube => panic!("TODO: I haven't checked the size of this yet"),
            GLSLTYPE::Texture2D => panic!("TODO: I haven't checked the size of this yet"),
            GLSLTYPE::Texture2DArray => panic!("TODO: I haven't checked the size of this yet"),
        }
    }
}

impl fmt::Display for GLSLTYPE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GLSLTYPE::Bool => write!(f, "bool"),
            GLSLTYPE::Float => write!(f, "float"),
            GLSLTYPE::Int => write!(f, "int"),
            GLSLTYPE::Uint => write!(f, "uint"),
            // we have this as a vec1 to know it is an array of floats but glsl only has float
            GLSLTYPE::Vec1 => write!(f, "float"),
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
            GLSLTYPE::SamplerShadow => write!(f, "samplerShadow"),
            GLSLTYPE::TextureCube => write!(f, "textureCube"),
            GLSLTYPE::Texture2D => write!(f, "texture2D"),
            GLSLTYPE::Texture2DArray => write!(f, "texture2DArray"),
        }
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
    (vec1) => {
        pipeline::shared::GLSLTYPE::Vec1
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
    (samplerShadow) => {
        pipeline::shared::GLSLTYPE::SamplerShadow
    };
    (textureCube) => {
        pipeline::shared::GLSLTYPE::TextureCube
    };
    (texture2D) => {
        pipeline::shared::GLSLTYPE::Texture2D
    };
    (texture2DArray) => {
        pipeline::shared::GLSLTYPE::Texture2DArray
    };
}


//todo why do I have this again?
pub const fn array_type(gtype: GLSLTYPE, depth: i64) -> GLSLTYPE {
    if depth == 1 {
        match gtype {
            GLSLTYPE::Float => GLSLTYPE::ArrayFloat,
            GLSLTYPE::Int => GLSLTYPE::ArrayInt,
            GLSLTYPE::Uint => GLSLTYPE::ArrayUint,
            GLSLTYPE::Vec1 => GLSLTYPE::ArrayVec2,
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
    string_compare(p, "gl_VertexID")
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
    pub group: Option<&'static str>,
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
    ( $([$($group:ident)? [$($qualifier:tt)*] $type:ident $($brack:tt)*] $param:ident;)*
      {$($tt:tt)*}) =>
      {
        {
            const S : &[pipeline::shared::PARAMETER] = &[$(
                pipeline::shared::PARAMETER{qual:&[$(qualifying!($qualifier)),*],
                                            gtype:pipeline::shared::array_type(typing!($type), count_brackets!($($brack)*)),
                                            name:stringify!($param),
                                            group:{let mut x : Option<&'static str> = None; $(x = Some(stringify!($group)); )? x},
                                        }),*];


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
            (S, B)
        }
      };
    }

#[macro_export]
macro_rules! my_shader {
    ($name:tt = {$($tt:tt)*}) =>
        {
        eager::eager_macro_rules! { $eager_1
            #[macro_export]
            macro_rules! $name{
                ()=>{$($tt)*};
            }
        }}
}
