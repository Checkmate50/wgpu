pub use self::shared::{
    can_pipe, compile_shader, process_body, ready_to_run, OutProgramBindings, ProgramBindings,
    BINDING, GLSLTYPE, QUALIFIER, array_type, new_bind_scope,
};

pub mod shared {
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
    pub fn compile_shader(
        contents: String,
        shader: ShaderType,
        device: &wgpu::Device,
    ) -> ShaderModule {
        // Convert our shader(in GLSL) to SPIR-V format
        // https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
        /*         print!("{}", contents);
        print!("\n\n"); */
        let mut vert_file = glsl_to_spirv::compile(&contents, shader)
            .unwrap_or_else(|_| panic!("{}: {}", "You gave a bad shader source", contents));
        let mut vs = Vec::new();
        vert_file
            .read_to_end(&mut vs)
            .expect("Somehow reading the file got interrupted");
        // Take the shader, ...,  and return
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap())
    }

    const fn string_compare(string1: &str, string2: &str) -> bool {
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
        return true;
    }

    pub const fn new_bind_scope(
        bind_context: &'static [&'static str; 32],
        bind_name: &'static str,
    ) -> ([&'static str; 32], bool) {
        let mut acc = 0;
        let mut found_it = false;
        let mut new_bind_context = [""; 32];
        while acc < 32 {
            if string_compare(bind_context[acc], bind_name) {
                found_it = true;
            } else {
                new_bind_context[acc] = bind_context[acc];
            }
            acc += 1;
        }
        (new_bind_context, found_it)
    }

    pub const fn ready_to_run(bind_context: [&'static str; 32]) -> bool {
        let mut acc = 0;
        while acc < 32 {
            if !string_compare(bind_context[acc], "") {
                return false;
            }
            acc += 1;
        }
        true
    }

    const fn params_contain_string(list_of_names: &[&str; 32], name: &str) -> bool {
        let mut acc = 0;
        while acc < 32 {
            if string_compare(list_of_names[acc], name) {
                return true;
            }
            acc += 1;
        }
        false
    }

    pub const fn can_pipe(s_out: &[&str; 32], s_in: &[&str; 32]) -> bool {
        let mut acc = 0;
        while acc < 32 {
            if !string_compare(s_out[acc], "") {
                if !params_contain_string(s_in, s_out[acc]) {
                    return false;
                }
            }
            acc += 1;
        }
        true
    }

    #[derive(Debug)]
    pub struct ProgramBindings {
        pub bindings: Vec<BINDING>,
    }

    #[derive(Debug)]
    pub struct OutProgramBindings {
        pub bindings: Vec<BINDING>,
    }

    #[derive(Debug)]
    pub struct BINDING {
        pub binding_number: u32,
        pub name: String,
        pub data: Option<wgpu::Buffer>,
        pub size: Option<u64>,
        pub gtype: GLSLTYPE,
        pub qual: Vec<QUALIFIER>,
    }

    #[derive(Debug, Clone, PartialEq)]
    #[allow(dead_code)]
    pub enum GLSLTYPE {
        Int,
        Uint,
        Float,
        Vec2,
        Vec3,
        ArrayInt,
        ArrayUint,
        ArrayFloat,
        ArrayVec2,
        ArrayVec3,
    }

    impl fmt::Display for GLSLTYPE {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                GLSLTYPE::Float => write!(f, "float"),
                GLSLTYPE::Int => write!(f, "int"),
                GLSLTYPE::Uint => write!(f, "uint"),
                GLSLTYPE::Vec2 => write!(f, "vec2"),
                GLSLTYPE::Vec3 => write!(f, "vec3"),
                GLSLTYPE::ArrayInt => write!(f, "int[]"),
                GLSLTYPE::ArrayUint => write!(f, "uint[]"),
                GLSLTYPE::ArrayFloat => write!(f, "float[]"),
                GLSLTYPE::ArrayVec2 => write!(f, "vec2[]"),
                GLSLTYPE::ArrayVec3 => write!(f, "vec3[]"),
            }
        }
    }

    #[macro_export]
    macro_rules! typing {
        (uint) => {
            shared::GLSLTYPE::Uint
        };
        (int) => {
            shared::GLSLTYPE::Int
        };
        (float) => {
            shared::GLSLTYPE::Float
        };
        (vec2) => {
            shared::GLSLTYPE::Vec2
        };
        (vec3) => {
            shared::GLSLTYPE::Vec3
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
        // opengl compute shaders don't have in and out variables so these are purely to try and interface at this library level
        IN,
        OUT,
        LOOP,
    }

    #[macro_export]
    macro_rules! qualifying {
        (buffer) => {
            shared::QUALIFIER::BUFFER
        };
        (uniform) => {
            shared::QUALIFIER::UNIFORM
        };
        (in) => {
            shared::QUALIFIER::IN
        };
        (out) => {
            shared::QUALIFIER::OUT
        };
        (loop) => {
            shared::QUALIFIER::LOOP
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
        ($brack:tt $($rest:tt)*) => {1 + count_brackets!($($rest)*)};
    }
}
