use wgpu::ShaderModule;

use glsl_to_spirv::ShaderType;

use std::fs::File;
use std::io::Read;
use std::path::Path;

// Read in a given file that should be a certain shader type and create a shader module out of it
fn compile_shader(file_name: &str, shader: ShaderType, device: &wgpu::Device) -> ShaderModule {
    assert!((Path::new(file_name)).exists());
    let mut file = File::open(file_name).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    // Convert our shader(in GLSL) to SPIR-V format
    // https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
    let mut vert_file = glsl_to_spirv::compile(&contents, shader)
        .unwrap_or_else(|_| panic!("{}: {}", "You gave a bad shader source", contents));
    let mut vs = Vec::new();
    vert_file
        .read_to_end(&mut vs)
        .expect("Somehow reading the file got interrupted");
    // Take the shader, ...,  and return
    device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap())
}

pub async fn compile(
    device: &wgpu::Device,
    vertex: &str,
    fragment: &str,
) -> wgpu::RenderPipeline {
    // the adapter is the handler to the physical graphics unit

    // todo vertex shader -> string + main entry point + descriptors...

    // Our compiled vertex shader
    let vs_module = compile_shader(vertex, ShaderType::Vertex, &device);

    // Our compiled fragment shader
    let fs_module = compile_shader(fragment, ShaderType::Fragment, &device);

    // todo I need to somehow analyze this from the vertex shader
    // Basically identifying attributes
    let vertex_buffer_desc = wgpu::VertexBufferDescriptor {
        stride: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        step_mode: wgpu::InputStepMode::Vertex,
        // If you have a struct that specifies your vertex, this is a 1 to 1 mapping of that struct
        attributes: &[
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // This is our connection to shader.vert
                shader_location: 0,
                format: wgpu::VertexFormat::Float3,
            },
            wgpu::VertexAttributeDescriptor {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                // This is our connection to shader.vert
                shader_location: 1,
                format: wgpu::VertexFormat::Float,
            },
        ],
    };

    // todo Set up the bindings for the pipeline
    // Basically uniforms

    // Create a layout for our bindings
    // If we had textures we would use this to lay them out
    /*     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
        bindings: &[],
        label: None,
    }); */

    // Bind no values to none of the bindings.
    // Use for something like textures
    /*     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[],
            label: None,
        });
    */
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        // If we supply a binding layout that is empty then everything crashes since the shaders use a layout
        // This layout is specifically used to specify textures often for the fragment stage
        bind_group_layouts: &[],
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
            vertex_buffers: &[vertex_buffer_desc],
        },
        // Number of samples to use per pixel(Use more than one for some fancy multisampling)
        sample_count: 1,
        // Use all available samples(This is a bitmask)
        sample_mask: !0,
        // Create a mask using the alpha values for each pixel and combine it with the sample mask to limit what samples are used
        alpha_to_coverage_enabled: false,
    });
    return render_pipeline;
}

//fn bind(program, value, location){}

pub fn with<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    frame: &'a wgpu::SwapChainOutput,
    render_pipeline: &'a wgpu::RenderPipeline,
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
            clear_color: wgpu::Color::WHITE,
        }],
        depth_stencil_attachment: None,
    });

    // The order must be set_pipeline -> set a bind_group if needed -> set a vertex buffer -> set an index buffer -> do draw
    // Otherwise we crash out
    rpass.set_pipeline(&render_pipeline);
    return rpass;
}

// todo Bind different variables individually
pub fn bind_vertex<'a>(rpass: &mut wgpu::RenderPass<'a>, vertex_buffer: &'a wgpu::Buffer) {
    rpass.set_vertex_buffer(0, &vertex_buffer, 0, 0);
}


// todo other types of bind functions

pub fn draw(
    rpass: &mut wgpu::RenderPass,
    vertices: core::ops::Range<u32>,
    instances: core::ops::Range<u32>,
) {
    wgpu::RenderPass::draw(rpass, vertices, instances);
}

// todo create a proper program struct that holds the render pass and the draw function
pub fn graphics_run(rpass: &mut wgpu::RenderPass) {
    draw(rpass, 0..3, 0..1);
}

// We wanted to somehow include that this will loop so we can try to do some kind of loop operation
// (T[] -> T)
//param attribute(x : T[]) : T = #loop <glsl x>

//param uniform(x : T) : T = <glsl x>

// (unit -> int)
//param constant<gl_VertexID>() : int = <glsl gl_VertexID>
//param constant<...>() : T = <glsl ...>
