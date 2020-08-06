pub use self::shared::{
    array_type, bind_float, bind_fvec, bind_fvec2, bind_vec, bind_vec2, bind_vec3, can_pipe,
    check_gl_builtin_type, compile_shader, has_in_qual, has_out_qual, is_gl_builtin,
    new_bind_scope, process_body, ready_to_run, string_compare, Bindings, OutProgramBindings,
    Program, ProgramBindings, BINDING, GLSLTYPE, PARAMETER, QUALIFIER,
};

pub mod shared {
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

    #[macro_export]
    macro_rules! update_bind_context {
        ($bind_context:tt, $bind_name:tt) => {{
            const BIND_CONTEXT: ([&str; 32], bool) = new_bind_scope(&$bind_context, $bind_name);
            const_assert!(BIND_CONTEXT.1);
            BIND_CONTEXT.0
        }};
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

    pub trait Bindings {
        fn clone(&self) -> Self;
    }

    #[derive(Debug)]
    pub struct ProgramBindings {
        pub bindings: Vec<BINDING>,
    }

    impl Bindings for ProgramBindings {
        fn clone(&self) -> ProgramBindings {
            ProgramBindings {
                bindings: new_bindings(&self.bindings),
            }
        }
    }

    #[derive(Debug)]
    pub struct OutProgramBindings {
        pub bindings: Vec<BINDING>,
    }
    impl Bindings for OutProgramBindings {
        fn clone(&self) -> OutProgramBindings {
            OutProgramBindings {
                bindings: new_bindings(&self.bindings),
            }
        }
    }

    fn new_bindings(bindings: &Vec<BINDING>) -> Vec<BINDING> {
        let mut new = Vec::new();

        for i in bindings.iter() {
            new.push(BINDING {
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

    #[derive(Debug)]
    pub struct BINDING {
        pub binding_number: u32,
        pub name: String,
        pub data: Option<wgpu::Buffer>,
        pub length: Option<u64>,
        pub gtype: GLSLTYPE,
        pub qual: Vec<QUALIFIER>,
    }

    fn bind<'a>(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        data: &'a [u8],
        length: u64,
        acceptable_types: Vec<GLSLTYPE>,
        name: String,
    ) {
        let binding = match bindings.bindings.iter().position(|x| x.name == name) {
            Some(x) => &mut bindings.bindings[x],
            None => {
                let x = out_bindings
                    .bindings
                    .iter()
                    .position(|x| x.name == name)
                    .expect("We couldn't find the binding");
                &mut out_bindings.bindings[x]
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

    pub fn bind_vec(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        numbers: &Vec<u32>,
        name: String,
    ) {
        bind(
            program,
            bindings,
            out_bindings,
            numbers.as_slice().as_bytes(),
            numbers.len() as u64,
            vec![GLSLTYPE::ArrayInt, GLSLTYPE::ArrayUint],
            name,
        )
    }

    pub fn bind_fvec(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        numbers: &Vec<f32>,
        name: String,
    ) {
        bind(
            program,
            bindings,
            out_bindings,
            numbers.as_slice().as_bytes(),
            numbers.len() as u64,
            vec![GLSLTYPE::Float, GLSLTYPE::ArrayFloat],
            name,
        )
    }

    pub fn bind_vec2(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        vecs: &Vec<Vec<f32>>,
        name: String,
    ) {
        let numbers: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        if numbers.len() % 2 != 0 {
            panic!("Your trying to bind to vec to but your not giving a vector that can be split into 2's")
        }
        bind(
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
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        numbers: &Vec<f32>,
        name: String,
    ) {
        if numbers.len() % 2 != 0 {
            panic!("Your trying to bind to vec to but your not giving a vector that can be split into 2's")
        }
        bind(
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
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        vecs: &Vec<Vec<f32>>,
        name: String,
    ) {
        let numbers: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        if numbers.len() % 3 != 0 {
            panic!("Your trying to bind to vec to but your not giving a vector that can be split into 3's")
        }
        bind(
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
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        mat: cgmath::Matrix4<f32>,
        name: String,
    ) {
        bind(
            program,
            bindings,
            out_bindings,
            &cgmath::conv::array4x4(mat).as_bytes(),
            1 as u64,
            vec![GLSLTYPE::Mat4],
            name,
        )
    }

    pub fn bind_float(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        numbers: &f32,
        name: String,
    ) {
        bind(
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
            }
        }
    }

    #[macro_export]
    macro_rules! typing {
        (bool) => {
            shared::GLSLTYPE::Bool
        };
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
        (uvec3) => {
            shared::GLSLTYPE::Uvec3
        };
        (vec3) => {
            shared::GLSLTYPE::Vec3
        };
        (vec4) => {
            shared::GLSLTYPE::Vec4
        };
        (mat4) => {
            shared::GLSLTYPE::Mat4
        };
        (sampler) => {
            shared::GLSLTYPE::Sampler
        };
        (textureCube) => {
            shared::GLSLTYPE::TextureCube
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
            shared::QUALIFIER::BUFFER
        };
        (uniform) => {
            shared::QUALIFIER::UNIFORM
        };
        (vertex) => {
            shared::QUALIFIER::VERTEX
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
            const S : &[shared::PARAMETER] = &[$(
                shared::PARAMETER{qual:&[$(qualifying!($qualifier)),*],
                                                      gtype:shared::array_type(typing!($type), count_brackets!($($brack)*)),
                                                      name:stringify!($param)}),*];


            const B: &'static str = munch_body!($($tt)*);

            let mut INBINDCONTEXT  = [""; 32];
            let mut OUTBINDCONTEXT = [""; 32];
            let mut acc = 0;
            while acc < 32 {
                if acc < S.len() {
                    if !is_gl_builtin(S[acc].name){
                    if shared::has_in_qual(S[acc].qual) {
                        INBINDCONTEXT[acc] = S[acc].name;
                    }
                    if shared::has_out_qual(S[acc].qual) {
                        OUTBINDCONTEXT[acc] = S[acc].name;
                    }}
                }
                acc += 1;
            }
            (S, B, INBINDCONTEXT, OUTBINDCONTEXT)
        }
      };
    }
}
