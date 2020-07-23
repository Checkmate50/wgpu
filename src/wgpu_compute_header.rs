pub use self::wgpu_compute_header::{
    array_type, bind_float, bind_vec, bind_vec2, can_pipe, compile, has_in_qual, has_out_qual,
    new_bind_scope, pipe, read_uvec, read_fvec, ready_to_run, run, GLSLTYPE, PARAMETER, QUALIFIER, SHADER,
};

pub mod wgpu_compute_header {
    use wgpu::ShaderModule;

    use glsl_to_spirv::ShaderType;

    use std::io::Read;

    use std::fmt;

    use std::collections::HashMap;

    use std::convert::TryInto;

    use regex::Regex;

    use zerocopy::AsBytes as _;

    // Remove spaces between tokens that should be one token
    // Strip off the starting and ending { }
    fn process_body(body: &str) -> String {
        let plus = Regex::new(r"\+(\n| )*\+").unwrap();
        let define = Regex::new(r"\#(\n| )*define").unwrap();
        //println!("{:?}", body);
        let in_progress = body.strip_prefix("{").unwrap().strip_suffix("}").unwrap();
        let plus_corrected = plus.replace_all(in_progress, "++").into_owned();
        define.replace_all(&plus_corrected, "#define").into_owned()
    }

    fn stringify_shader(s: &SHADER, b: &ProgramBindings, b_out: &OutProgramBindings) -> String {
        let mut buffer = Vec::new();
        for i in &b.bindings[..] {
            buffer.push(format!(
                "layout(binding = {}) {} BINDINGS{} {{\n",
                i.binding_number,
                if i.qual.contains(&QUALIFIER::BUFFER) {
                    "buffer"
                } else if i.qual.contains(&QUALIFIER::UNIFORM) {
                    "uniform"
                } else {panic!("You are trying to do something with something that isn't a buffer or uniform")},
                i.binding_number
            ));

            buffer.push(i.gtype.to_string() + " " + &i.name + ";\n");
            buffer.push("};\n".to_string());
        }
        for i in &b_out.bindings[..] {
            if i.qual.contains(&QUALIFIER::BUFFER) {
                buffer.push(format!(
                    "layout(binding = {}) buffer BINDINGS{} {{\n",
                    i.binding_number, i.binding_number
                ));

                buffer.push(i.gtype.to_string() + " " + &i.name + ";\n");
                buffer.push("};\n".to_string());
            }
        }
        format!(
            //todo figure out how to use a non-1 local size
            "#version 450\nlayout(local_size_x = 1) in;\n{}\n\n{}\n",
            buffer.join(""),
            process_body(s.body)
        )
    }

    // Read in a given file that should be a certain shader type and create a shader module out of it
    fn compile_shader(contents: String, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
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

    #[derive(Debug)]
    pub struct BINDING {
        binding_number: u32,
        name: String,
        data: Option<wgpu::Buffer>,
        size: Option<u64>,
        gtype: GLSLTYPE,
        qual: Vec<QUALIFIER>,
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
    ) -> (wgpu::BindGroupLayout, ProgramBindings, OutProgramBindings) {
        let mut binding_struct = Vec::new();
        let mut binding_number = 0;
        let mut out_binding_struct = Vec::new();
        for i in &compute.params[..] {
            // Bindings that are kept between runs
            if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                binding_struct.push(BINDING {
                    binding_number: binding_number,
                    name: i.name.to_string(),
                    data: None,
                    size: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                binding_number += 1;
            // Bindings that are invalidated after a run
            } else if i.qual.contains(&QUALIFIER::OUT) {
                out_binding_struct.push(BINDING {
                    binding_number: binding_number,
                    name: i.name.to_string(),
                    data: None,
                    size: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                binding_number += 1;
            }
        }

        // Create a layout for our bindings
        // If we had textures we would use this to lay them out

        let mut bind_entry = Vec::new();

        for i in &binding_struct {
            bind_entry.push(wgpu::BindGroupLayoutEntry {
                binding: i.binding_number,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: if i.qual.contains(&QUALIFIER::UNIFORM) {
                    wgpu::BindingType::UniformBuffer { dynamic: false }
                } else {
                    wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                    }
                },
            });
        }

        for i in &out_binding_struct {
            bind_entry.push(wgpu::BindGroupLayoutEntry {
                binding: i.binding_number,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: if i.qual.contains(&QUALIFIER::UNIFORM) {
                    wgpu::BindingType::UniformBuffer { dynamic: false }
                } else {
                    wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                    }
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
            OutProgramBindings {
                bindings: out_binding_struct,
            },
        );
    }

    pub struct PROGRAM {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    pub async fn compile(compute: &SHADER) -> (PROGRAM, ProgramBindings, OutProgramBindings) {
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
        let (bind_group_layout, program_bindings, out_program_bindings) =
            create_bindings(&compute, &device);

        let cs_module = compile_shader(
            stringify_shader(&compute, &program_bindings, &out_program_bindings),
            ShaderType::Compute,
            &device,
        );

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
            out_program_bindings,
        );
    }

    fn bind<'a>(
        program: &PROGRAM,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        data: &'a [u8],
        size: u64,
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
            panic!(
                "The type of the value you provided is not what was expected, {:?}",
                &binding.gtype
            );
        }

        let buffer = program.device.create_buffer_with_data(
            data,
            if binding.qual.contains(&QUALIFIER::UNIFORM) {
                wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST
            } else {
                wgpu::BufferUsage::MAP_READ
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_SRC
            },
        );

        binding.data = Some(buffer);
        binding.size = Some(size);
    }

    pub fn bind_vec(
        program: &PROGRAM,
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
            (numbers.len() * std::mem::size_of::<u32>()) as u64,
            vec![GLSLTYPE::ArrayInt, GLSLTYPE::ArrayUint],
            name,
        )
    }

    pub fn bind_vec2(
        program: &PROGRAM,
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
            (numbers.len() * std::mem::size_of::<u32>()) as u64,
            if numbers.len() == 2 {
                vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
            } else {
                vec![GLSLTYPE::ArrayVec2]
            },
            name,
        )
    }

    pub fn bind_float(
        program: &PROGRAM,
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
            std::mem::size_of::<f32>() as u64,
            vec![GLSLTYPE::Float],
            name,
        )
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

    pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
        cpass.dispatch(length, 1, 1);
    }

    fn buffer_map_setup<'a>(
        bindings: &'a ProgramBindings,
        out_bindings: &'a OutProgramBindings,
    ) -> HashMap<u32, &'a BINDING> {
        let mut buffer_map = HashMap::new();

        for i in bindings.bindings.iter() {
            buffer_map.insert(i.binding_number, i);
        }

        for i in out_bindings.bindings.iter() {
            buffer_map.insert(i.binding_number, i);
        }
        return buffer_map;
    }

    pub fn run(
        program: &PROGRAM,
        bindings: &ProgramBindings,
        mut out_bindings: OutProgramBindings,
    ) -> Vec<BINDING> {
        let mut encoder = program
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Look for a loop qualifier in bindings, if it isn't there, it must be in out_bindings or we just give 1
        // Use this to get the size that the program should run over
        let mut bind = bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::LOOP));

        if bind.is_none() {
            bind = out_bindings
                .bindings
                .iter()
                .find(|i| i.qual.contains(&QUALIFIER::LOOP));
        }
        let size = if bind.is_none() {
            1
        } else {
            bind.unwrap().size.unwrap()
        };

        for i in 0..(out_bindings.bindings.len()) {
            if !(out_bindings.bindings[i].qual.contains(&QUALIFIER::IN)) {
                out_bindings.bindings[i].data =
                    Some(program.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size,
                        usage: wgpu::BufferUsage::STORAGE
                            | wgpu::BufferUsage::MAP_READ
                            | wgpu::BufferUsage::COPY_DST
                            | wgpu::BufferUsage::COPY_SRC,
                    }));
                out_bindings.bindings[i].size = Some(size);
            }
        }

        let buffer_map = buffer_map_setup(bindings, &out_bindings);

        let mut empty_vec = Vec::new();

        {
            for i in 0..(buffer_map.len()) {
                let b = buffer_map.get(&(i as u32)).expect(&format!(
                    "I assumed all bindings would be buffers but I guess that has been invalidated"
                ));
                empty_vec.push(wgpu::Binding {
                    binding: b.binding_number,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &b
                            .data
                            .as_ref()
                            .expect(&format!("The binding of {} was not set", &b.name)),
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
            compute(&mut cpass, size as u32);
        }
        program.queue.submit(&[encoder.finish()]);

        out_bindings.bindings
    }

    pub async fn read_uvec(program: &PROGRAM, results: &Vec<BINDING>, name: &str) -> Vec<u32> {
        for i in results.iter() {
            if i.name == name {
                let result_buffer = i.data.as_ref().unwrap();
                let buffer_future = result_buffer.map_read(0, i.size.unwrap());
                program.device.poll(wgpu::Maintain::Wait);

                if let Ok(mapping) = buffer_future.await {
                    let x: Vec<u32> = mapping
                        .as_slice()
                        .chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect();
                    return x;
                } else {
                    panic!("failed to run compute on gpu!");
                }
            }
        }
        panic!(
            "We didn't find the binding you were looking to read from: {}",
            name
        )
    }

    pub async fn read_fvec(program: &PROGRAM, results: &Vec<BINDING>, name: &str) -> Vec<f32> {
        for i in results.iter() {
            if i.name == name {
                let result_buffer = i.data.as_ref().unwrap();
                let buffer_future = result_buffer.map_read(0, i.size.unwrap());
                program.device.poll(wgpu::Maintain::Wait);

                if let Ok(mapping) = buffer_future.await {
                    let x: Vec<f32> = mapping
                        .as_slice()
                        .chunks_exact(4)
                        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                        .collect();
                    return x;
                } else {
                    panic!("failed to run compute on gpu!");
                }
            }
        }
        panic!(
            "We didn't find the binding you were looking to read from: {}",
            name
        )
    }

    pub fn pipe(
        program: &PROGRAM,
        mut in_bindings: ProgramBindings,
        mut out_bindings: OutProgramBindings,
        result_vec: Vec<BINDING>,
    ) -> Vec<BINDING> {
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
            binding.size = Some(i.size.unwrap());
        }

        run(program, &in_bindings, out_bindings)
    }

    // TODO
    // Develop the syntax for binding a variable
    //     Scoping for bind operations?
    //
    // HOLD
    // const shader for statically checking names?
    // Use specification to try an create more static checking?
    //
    // Maybe work on result and getting outputs
    //
    // Syntax/annotations to get rid of magic variables in shader like gl_GlobalInvocationID.x?
    //
    // Find a realistic compute shader and implement it


    // End to end boids examples

    // Use github issues for project management

    #[derive(Debug, Clone, PartialEq)]
    #[allow(dead_code)]
    pub enum GLSLTYPE {
        Int,
        Uint,
        Float,
        Vec2,
        ArrayInt,
        ArrayUint,
        ArrayFloat,
        ArrayVec2,
    }

    impl fmt::Display for GLSLTYPE {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                GLSLTYPE::Float => write!(f, "float"),
                GLSLTYPE::Int => write!(f, "int"),
                GLSLTYPE::Uint => write!(f, "uint"),
                GLSLTYPE::Vec2 => write!(f, "vec2"),
                GLSLTYPE::ArrayInt => write!(f, "int[]"),
                GLSLTYPE::ArrayUint => write!(f, "uint[]"),
                GLSLTYPE::ArrayFloat => write!(f, "float[]"),
                GLSLTYPE::ArrayVec2 => write!(f, "vec2[]"),
            }
        }
    }

    #[macro_export]
    macro_rules! typing {
        (uint) => {
            wgpu_compute_header::GLSLTYPE::Uint
        };
        (int) => {
            wgpu_compute_header::GLSLTYPE::Int
        };
        (float) => {
            wgpu_compute_header::GLSLTYPE::Float
        };
        (vec2) => {
            wgpu_compute_header::GLSLTYPE::Vec2
        };
    }

    pub const fn array_type(gtype: GLSLTYPE, depth: i64) -> GLSLTYPE {
        if depth == 1 {
            match gtype {
                GLSLTYPE::Float => GLSLTYPE::ArrayFloat,
                GLSLTYPE::Int => GLSLTYPE::ArrayInt,
                GLSLTYPE::Uint => GLSLTYPE::ArrayUint,
                GLSLTYPE::Vec2 => GLSLTYPE::ArrayVec2,
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
            wgpu_compute_header::QUALIFIER::BUFFER
        };
        (uniform) => {
            wgpu_compute_header::QUALIFIER::UNIFORM
        };
        (in) => {
            wgpu_compute_header::QUALIFIER::IN
        };
        (out) => {
            wgpu_compute_header::QUALIFIER::OUT
        };
        (loop) => {
            wgpu_compute_header::QUALIFIER::LOOP
        };
    }

    #[derive(Debug)]
    pub struct PARAMETER {
        pub qual: &'static [QUALIFIER],
        pub gtype: GLSLTYPE,
        pub name: &'static str,
    }

    #[derive(Debug)]
    pub struct SHADER {
        pub params: &'static [PARAMETER],
        pub body: &'static str,
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
            const S : &[wgpu_compute_header::PARAMETER] = &[$(
                wgpu_compute_header::PARAMETER{qual:&[$(qualifying!($qualifier)),*],
                                                      gtype:wgpu_compute_header::array_type(typing!($type), count_brackets!($($brack)*)),
                                                      name:stringify!($param)}),*];


            const B: &'static str = munch_body!($($tt)*);

            let mut INBINDCONTEXT  = [""; 32];
            let mut OUTBINDCONTEXT = [""; 32];
            let mut acc = 0;
            while acc < 32 {
                if acc < S.len() {
                    if wgpu_compute_header::has_in_qual(S[acc].qual){
                        INBINDCONTEXT[acc] = S[acc].name;
                    }
                    if wgpu_compute_header::has_out_qual(S[acc].qual){
                        OUTBINDCONTEXT[acc] = S[acc].name;
                    }
                }
                acc += 1;
            }
            (wgpu_compute_header::SHADER{params:S, body:B}, INBINDCONTEXT, OUTBINDCONTEXT)
        }
    };
}
}
