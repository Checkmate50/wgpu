pub use self::wgpu_graphics_header::{
    compile_buffer, graphics_compile, graphics_pipe, graphics_run, valid_fragment_shader,
    valid_vertex_shader, GraphicsProgram, GraphicsShader,
};

pub mod wgpu_graphics_header {
    use glsl_to_spirv::ShaderType;

    use std::collections::HashMap;

    use winit::window::Window;

    use crate::shared::{
        check_gl_builtin_type, compile_shader, has_out_qual, process_body, string_compare,
        OutProgramBindings, Program, ProgramBindings, BINDING, GLSLTYPE, PARAMETER, QUALIFIER,
    };

    pub struct GraphicsProgram {
        pub surface: wgpu::Surface,
        pub device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::RenderPipeline,
        // bind_group_layout: wgpu::BindGroupLayout,
    }

    impl Program for GraphicsProgram {
        fn get_device(&self) -> &wgpu::Device {
            &self.device
        }
    }

    impl Program for &GraphicsProgram {
        fn get_device(&self) -> &wgpu::Device {
            &self.device
        }
    }

    fn stringify_shader(
        s: &GraphicsShader,
        b: &ProgramBindings,
        b_out: &OutProgramBindings,
    ) -> String {
        let mut buffer = Vec::new();
        for i in &b.bindings[..] {
            if i.name != "gl_Position" {
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
                        panic!(
                            "You are trying to do something with something that isn't an in or out"
                        )
                    },
                    i.gtype,
                    i.name
                ));
            }
        }
        for i in &b_out.bindings[..] {
            if i.name != "gl_Position" {
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
                        panic!(
                            "You are trying to do something with something that isn't an in or out"
                        )
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

    fn create_bindings<'a>(
        vertex: &GraphicsShader,
        fragment: &GraphicsShader,
    ) -> (
        ProgramBindings,
        OutProgramBindings,
        ProgramBindings,
        OutProgramBindings,
    ) {
        let mut vertex_binding_struct = Vec::new();
        let mut vertex_out_binding_struct = Vec::new();
        let mut fragment_binding_struct = Vec::new();
        let mut fragment_out_binding_struct = Vec::new();
        let mut vertex_binding_number = 0;
        let mut vertex_to_fragment_binding_number = 0;
        let mut vertex_to_fragment_map = HashMap::new();
        let mut fragment_out_binding_number = 0;
        for i in &vertex.params[..] {
            if !check_gl_builtin_type(i.name, &i.gtype) {
                // Bindings that are kept between runs
                if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                    vertex_binding_struct.push(BINDING {
                        binding_number: vertex_binding_number,
                        name: i.name.to_string(),
                        data: None,
                        length: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    vertex_binding_number += 1;
                // Bindings that are invalidated after a run
                } else if !i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                    vertex_out_binding_struct.push(BINDING {
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
                    panic!("TODO We currently don't support both in and out qualifiers for vertex/fragment shaders")
                }
            }
        }

        for i in &fragment.params[..] {
            if !check_gl_builtin_type(i.name, &i.gtype) {
                // Bindings that are kept between runs
                if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                    fragment_binding_struct.push(BINDING {
                        binding_number: vertex_to_fragment_map.get(i.name).unwrap().clone(),
                        name: i.name.to_string(),
                        data: None,
                        length: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                // Bindings that are invalidated after a run
                } else if !i.qual.contains(&QUALIFIER::IN) && i.qual.contains(&QUALIFIER::OUT) {
                    fragment_out_binding_struct.push(BINDING {
                        binding_number: fragment_out_binding_number,
                        name: i.name.to_string(),
                        data: None,
                        length: None,
                        gtype: i.gtype.clone(),
                        qual: i.qual.to_vec(),
                    });
                    fragment_out_binding_number += 1;
                } else {
                    panic!("TODO We currently don't support both in and out qualifiers for vertex/fragment shaders")
                }
            }
        }

        return (
            ProgramBindings {
                bindings: vertex_binding_struct,
            },
            OutProgramBindings {
                bindings: vertex_out_binding_struct,
            },
            ProgramBindings {
                bindings: fragment_binding_struct,
            },
            OutProgramBindings {
                bindings: fragment_out_binding_struct,
            },
        );
    }

    pub async fn graphics_compile(
        vec_buffer: &mut [wgpu::VertexAttributeDescriptor; 32],
        window: &Window,
        vertex: &GraphicsShader,
        fragment: &GraphicsShader,
    ) -> (GraphicsProgram, ProgramBindings, OutProgramBindings) {
        // the adapter is the handler to the physical graphics unit

        // todo vertex shader -> string + main entry point + descriptors...

        // Create a surface to draw images on
        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                // Can specify Low/High power usage
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            // Map to Vulkan/Metal/Direct3D 12
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

        // The device manages the connection and resources of the adapter
        // The queue is a literal queue of tasks for the gpu
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            })
            .await;

        let (program_bindings1, out_program_bindings1, program_bindings2, out_program_bindings2) =
            create_bindings(&vertex, &fragment);

        for i in &program_bindings1.bindings[..] {
            vec_buffer[i.binding_number as usize] = wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: i.binding_number,
                format: if i.gtype == GLSLTYPE::Vec3 {
                    wgpu::VertexFormat::Float3
                } else {
                    wgpu::VertexFormat::Float
                },
            };
        }

        let mut vertex_binding_desc = Vec::new();
        let mut bind_entry = Vec::new();
        let mut are_bind_enties = false;

        for i in &program_bindings1.bindings[..] {
            if i.qual.contains(&QUALIFIER::UNIFORM) {
                bind_entry.push(wgpu::BindGroupLayoutEntry {
                    binding: i.binding_number,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false }, /* else {
                                                                                 wgpu::BindingType::StorageBuffer {
                                                                                     dynamic: false,
                                                                                     readonly: false,
                                                                                 }
                                                                             } */
                });
                are_bind_enties = true;
            } else {
                vertex_binding_desc.push(wgpu::VertexBufferDescriptor {
                    stride: (if i.gtype == GLSLTYPE::Vec3 {
                        std::mem::size_of::<[f32; 3]>()
                    } else {
                        std::mem::size_of::<f32>()
                    }) as wgpu::BufferAddress,
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

        for i in &out_program_bindings1.bindings {
            if i.qual.contains(&QUALIFIER::UNIFORM) && i.qual.contains(&QUALIFIER::IN) {
                bind_entry.push(wgpu::BindGroupLayoutEntry {
                    binding: i.binding_number,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false }, /* else {
                                                                                 wgpu::BindingType::StorageBuffer {
                                                                                     dynamic: false,
                                                                                     readonly: false,
                                                                                 }
                                                                             } */
                });
                are_bind_enties = true;
            }
        }

        // Our compiled vertex shader
        let vs_module = compile_shader(
            stringify_shader(vertex, &program_bindings1, &out_program_bindings1),
            ShaderType::Vertex,
            &device,
        );

        // Our compiled fragment shader
        let fs_module = compile_shader(
            stringify_shader(fragment, &program_bindings2, &out_program_bindings2),
            ShaderType::Fragment,
            &device,
        );

        let bind_group_layout =
            &[
                &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
                    bindings: if are_bind_enties { &bind_entry } else { &[] },
                    label: None,
                }),
            ];

        // Bind no values to none of the bindings.
        // Use for something like textures
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: if are_bind_enties {
                bind_group_layout
            } else {
                &[]
            },
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
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            // Use Triangles
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
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
            // We can add an optional stencil descriptor which allows for effects that you would see in Microsoft Powerpoint like fading/swiping to the next slide
            depth_stencil_state: None,
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
        (
            GraphicsProgram {
                pipeline: render_pipeline,
                device,
                queue,
                surface,
            },
            program_bindings1,
            out_program_bindings1,
        )
    }

    /* pub fn bind_texture(
        program: &dyn Program,
        bindings: &mut ProgramBindings,
        out_bindings: &mut OutProgramBindings,
        numbers: &Vec<u32>,
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

        if ![GLSLTYPE::TextureCube].contains(&binding.gtype) {
            println!("{:?}", &binding.name);
            println!("{:?}", GLSLTYPE::TextureCube);
            panic!(
                "The type of the value you provided is not what was expected, {:?}",
                &binding.gtype
            );
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: SKYBOX_FORMAT,
            dimension: wgpu::TextureViewDimension::Cube,
            aspect: wgpu::TextureAspect::default(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_layer_count: 6,
        });

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
    } */

    pub fn draw(
        rpass: &mut wgpu::RenderPass,
        vertices: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
    ) {
        rpass.draw(vertices, instances);
    }

    // todo create a proper program struct that holds the render pass and the draw function
    pub fn graphics_run(
        program: &GraphicsProgram,
        bindings: &ProgramBindings,
        out_bindings: OutProgramBindings,
        swap_chain: &mut wgpu::SwapChain,
    ) {
        let frame = swap_chain
            .get_next_texture()
            .expect("Timeout when acquiring next swap chain texture");

        let mut encoder = program
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let bind = bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::VERTEX));

        let verts: u32 = if bind.is_none() {
            3
        } else {
            bind.unwrap().length.unwrap() as u32
        };

        let bind = bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::LOOP));

        let instances: u32 = if bind.is_none() {
            1
        } else {
            bind.unwrap().length.unwrap() as u32
        };

        {
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
                    clear_color: wgpu::Color::BLACK,
                }],
                depth_stencil_attachment: None,
            });

            // The order must be set_pipeline -> set a bind_group if needed -> set a vertex buffer -> set an index buffer -> do draw
            // Otherwise we crash out
            rpass.set_pipeline(&program.pipeline);

            for i in 0..(bindings.bindings.len()) {
                let b = bindings.bindings.get(i as usize).expect(&format!(
                    "I assumed all bindings would be buffers but I guess that has been invalidated"
                ));
                rpass.set_vertex_buffer(
                    b.binding_number,
                    &b.data
                        .as_ref()
                        .expect(&format!("The binding of {} was not set", &b.name)),
                    0,
                    0,
                );
            }

            for i in 0..(out_bindings.bindings.len()) {
                let b = out_bindings.bindings.get(i as usize).expect(&format!(
                    "I assumed all bindings would be buffers but I guess that has been invalidated"
                ));
                if b.qual.contains(&QUALIFIER::IN) {
                    rpass.set_vertex_buffer(
                        b.binding_number,
                        &b.data
                            .as_ref()
                            .expect(&format!("The binding of {} was not set", &b.name)),
                        0,
                        0,
                    );
                }
            }

            draw(&mut rpass, 0..verts, 0..instances);
        }

        // Do the rendering by saying that we are done and sending it off to the gpu
        program.queue.submit(&[encoder.finish()]);
    }

    pub fn graphics_pipe(
        program: &GraphicsProgram,
        mut in_bindings: ProgramBindings,
        mut out_bindings: OutProgramBindings,
        swap_chain: &mut wgpu::SwapChain,
        result_vec: Vec<BINDING>,
    ) {
        println!("starting");
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

        graphics_run(program, &in_bindings, out_bindings, swap_chain);
    }

    #[derive(Debug)]
    pub struct GraphicsShader {
        pub params: &'static [PARAMETER],
        pub body: &'static str,
    }

    pub const fn valid_vertex_shader(V: &GraphicsShader) -> bool {
        let mut acc = 0;
        while acc < V.params.len() {
            if string_compare(V.params[acc].name, "gl_Position") && has_out_qual(V.params[acc].qual)
            {
                if let GLSLTYPE::Vec4 = V.params[acc].gtype {
                    return true;
                }
            }
            acc = acc + 1;
        }
        false
    }

    pub const fn valid_fragment_shader(F: &GraphicsShader) -> bool {
        let mut acc = 0;
        while acc < F.params.len() {
            if string_compare(F.params[acc].name, "color") && has_out_qual(F.params[acc].qual) {
                if let GLSLTYPE::Vec4 = F.params[acc].gtype {
                    return true;
                }
            }
            acc = acc + 1;
        }
        false
    }

    #[macro_export]
    macro_rules! graphics_shader {
        ($($body:tt)*) => {{
            const S : (&[shared::PARAMETER], &'static str, [&str; 32], [&str; 32]) = shader!($($body)*);
            (wgpu_graphics_header::GraphicsShader{params:S.0, body:S.1}, S.2, S.3)
        }};
    }

    // This is a crazy hack that happens because of a couple things
    // -- I need to be able to create VertexAttributeDescriptors in compile and save a reference to them when creating the pipeline
    // -- I can't initialize an empty array
    // -- wgpu::VertexAttributeDescriptor is non-Copyable so doing the usual [desc; 32] doesn't work
    // -- I don't want to use unsafe code or at the moment use another library like arrayvec
    pub fn compile_buffer() -> [wgpu::VertexAttributeDescriptor; 32] {
        [
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float,
            },
        ]
    }
}
