#![feature(async_closure)]
#[macro_use]
extern crate pipeline;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use std::cell::RefCell;

// use for the shader! macro
pub use pipeline::shared;
pub use pipeline::wgpu_compute_header;
pub use pipeline::wgpu_graphics_header;

pub use pipeline::shared::{
    bind_float, bind_fvec2, bind_vec, bind_vec3, can_pipe, is_gl_builtin, new_bind_scope,
    ready_to_run, Bindings, OutProgramBindings, Program, ProgramBindings,
};
pub use pipeline::wgpu_compute_header::{
    compile, read_fvec3, run, ComputeBindings, ComputeProgram, ComputeShader, OutComputeBindings,
};
pub use pipeline::wgpu_graphics_header::{
    compile_buffer, graphics_compile, graphics_pipe, graphics_run, valid_fragment_shader,
    valid_vertex_shader, GraphicsBindings, GraphicsProgram, GraphicsShader, OutGraphicsBindings,
};

pub use static_assertions::const_assert;

fn execute_gpu(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();
    // qualifiers
    // buffer: is a buffer?
    // in: this parameter must be bound to before the program runs
    //     thus it can be read inside of the program scope
    // out: if this parameter is bound, it must be rebound after each iteration
    //      Only out variables can be mutated
    //      Only out variables can be read as a result of the program
    //      If out has been unassigned then an error is raised when it is read
    // loop: one or more of these loop annotations are required per program. Atm, the values bound is assumed to be of equal length and this gives the number of iterations(gl_GlobalInvocationID.x)
    //      the size of any out buffers that need to be created

    // TODO how to set a work group larger than 1?
    const BOIDS: (ComputeShader, [&str; 32], [&str; 32]) = compute_shader! {
        [[uniform in] float] deltaT;
        [[uniform in] float] rule1Distance;
        [[uniform in] float] rule2Distance;
        [[uniform in] float] rule3Distance;
        [[uniform in] float] rule1Scale;
        [[uniform in] float] rule2Scale;
        [[uniform in] float] rule3Scale;
        [[] uvec3] gl_GlobalInvocationID;

        [[buffer loop in] vec3[]] srcParticlePos;
        [[buffer loop in] vec3[]] srcParticleVel;
        [[buffer loop in out] vec3[]] trianglePos;
        [[buffer out] vec3[]] dstParticlePos;
        [[buffer out] vec3[]] dstParticleVel;

        {{
            void main() {
                // https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
                uint index = gl_GlobalInvocationID.x;

                /*     if (index >= NUM_PARTICLES) { return; } */
                vec2 vPos = srcParticlePos[index].xy;
                vec2 vVel = srcParticleVel[index].xy;
                vec2 cMass = vec2(0.0, 0.0);
                vec2 cVel = vec2(0.0, 0.0);
                vec2 colVel = vec2(0.0, 0.0);
                int cMassCount = 0;
                int cVelCount = 0;
                vec2 pos;
                vec2 vel;
                // TODO The iteration of the number of particles was set by a #define value
                for (int i = 0; i < 2; ++i) {
                    if (i == index) { continue; }
                    pos = srcParticlePos[i].xy;
                    vel = srcParticlePos[i].xy;
                    if (distance(pos, vPos) < rule1Distance) {
                        cMass += pos;
                        cMassCount++;
                    }
                    if (distance(pos, vPos) < rule2Distance) {
                        colVel -= (pos - vPos);
                    }
                    if (distance(pos, vPos) < rule3Distance) {
                        cVel += vel;
                        cVelCount++;
                    }
                }
                if (cMassCount > 0) {
                    cMass = cMass / cMassCount - vPos;
                }
                if (cVelCount > 0) {
                    cVel = cVel / cVelCount;
                }
                vVel += cMass * rule1Scale + colVel * rule2Scale + cVel * rule3Scale;
                // clamp velocity for a more pleasing simulation.
                vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
                // kinematic update
                vPos += vVel * deltaT;
                // Wrap around boundary
                if (vPos.x < -1.0) vPos.x = 1.0;
                if (vPos.x > 1.0) vPos.x = -1.0;
                if (vPos.y < -1.0) vPos.y = 1.0;
                if (vPos.y > 1.0) vPos.y = -1.0;
                    // Write back
                dstParticlePos[index] = vec3(vPos, 0.0);
                dstParticleVel[index] = vec3(vVel, 0.0);
            }
        }}
    };

    const VERTEX: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[loop in] vec3] dstParticlePos;
        [[loop in] vec3] dstParticleVel;
        [[vertex in] vec3] trianglePos;
        [[out] vec4] gl_Position;

        {{
            void main() {
                float angle = -atan(dstParticleVel.x, dstParticleVel.y);
                vec3 pos = vec3(trianglePos.x * cos(angle) - trianglePos.y * sin(angle),
                                trianglePos.x * sin(angle) + trianglePos.y * cos(angle), 0);
                gl_Position = vec4(pos + dstParticlePos, 1);
            }
        }}
    };

    const FRAGMENT: (GraphicsShader, [&str; 32], [&str; 32]) = graphics_shader! {
        [[out] vec4] color;

        {{
            void main() {
                color = vec4(1.0);
            }
        }}
    };

    const S: ComputeShader = BOIDS.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = BOIDS.1;
    const V: GraphicsShader = VERTEX.0;
    const NEXT_STARTING_CONTEXT: [&str; 32] = VERTEX.1;
    const F: GraphicsShader = FRAGMENT.0;

    let mut compile_buffer: [wgpu::VertexAttributeDescriptor; 32] = compile_buffer();

    let (program, template_bindings, template_out_bindings) =
        futures::executor::block_on(compile(&S));

    static_assertions::const_assert!(valid_vertex_shader(&V));
    static_assertions::const_assert!(valid_fragment_shader(&F));
    let (graphics_program, template_graphics_bindings, template_graphics_out_bindings) =
        futures::executor::block_on(graphics_compile(&mut compile_buffer, &window, &V, &F));

    // For drawing to window
    let sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // Window dimensions
        width: size.width,
        height: size.height,
        // Only update during the "vertical blanking interval"
        // As opposed to Immediate where it is possible to see visual tearing(where multiple frames are visible at once)
        present_mode: wgpu::PresentMode::Mailbox,
    };

    // A "chain" of buffers that we render on to the display
    let mut swap_chain = RefCell::new(
        graphics_program
            .device
            .create_swap_chain(&graphics_program.surface, &sc_desc),
    );

    let mut srcParticlePos = RefCell::new(vec![[0.5, 0.2, 0.0], [0.2, 0.1, 0.0]]);
    let mut srcParticleVel = RefCell::new(vec![[-0.1, -0.1, 0.0], [0.15, -0.12, 0.0]]);
    let triangle: Vec<[f32; 3]> = vec![[-0.01, -0.02, 0.0], [0.01, -0.02, 0.0], [0.00, 0.02, 0.0]];

    async fn draw(
        frame: &mut wgpu::SwapChain,
        program: &ComputeProgram,
        graphics_program: &GraphicsProgram,
        template_bindings: &ComputeBindings,
        template_out_bindings: &OutComputeBindings,
        template_graphics_bindings: &GraphicsBindings,
        template_graphics_out_bindings: &OutGraphicsBindings,
        srcParticlePos: &RefCell<Vec<[f32; 3]>>,
        srcParticleVel: &RefCell<Vec<[f32; 3]>>,
    ) -> () {
        let triangle: Vec<[f32; 3]> =
            vec![[-0.01, -0.02, 0.0], [0.01, -0.02, 0.0], [0.00, 0.02, 0.0]];
        let deltaT: f32 = 0.04;
        let rule1Distance: f32 = 0.3;
        let rule2Distance: f32 = 0.15;
        let rule3Distance: f32 = 0.05;
        let rule1Scale: f32 = 0.3;
        let rule2Scale: f32 = 0.1;
        let rule3Scale: f32 = 0.05;

        let mut bindings: ComputeBindings = template_bindings.clone();
        let mut out_bindings: OutComputeBindings = template_out_bindings.clone();
        let mut graphics_bindings = template_graphics_bindings.clone();
        let mut graphics_out_bindings = template_graphics_out_bindings.clone();

        const BIND_CONTEXT_1: [&str; 32] = update_bind_context!(STARTING_BIND_CONTEXT, "deltaT");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &deltaT,
            "deltaT".to_string(),
        );

        const BIND_CONTEXT_2: [&str; 32] = update_bind_context!(BIND_CONTEXT_1, "rule1Distance");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule1Distance,
            "rule1Distance".to_string(),
        );

        const BIND_CONTEXT_3: [&str; 32] = update_bind_context!(BIND_CONTEXT_2, "rule2Distance");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule2Distance,
            "rule2Distance".to_string(),
        );

        const BIND_CONTEXT_4: [&str; 32] = update_bind_context!(BIND_CONTEXT_3, "rule3Distance");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule3Distance,
            "rule3Distance".to_string(),
        );

        const BIND_CONTEXT_5: [&str; 32] = update_bind_context!(BIND_CONTEXT_4, "rule1Scale");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule1Scale,
            "rule1Scale".to_string(),
        );

        const BIND_CONTEXT_6: [&str; 32] = update_bind_context!(BIND_CONTEXT_5, "rule2Scale");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule2Scale,
            "rule2Scale".to_string(),
        );

        const BIND_CONTEXT_7: [&str; 32] = update_bind_context!(BIND_CONTEXT_6, "rule3Scale");
        bind_float(
            &program,
            &mut bindings,
            &mut out_bindings,
            &rule3Scale,
            "rule3Scale".to_string(),
        );

        const BIND_CONTEXT_8: [&str; 32] = update_bind_context!(BIND_CONTEXT_7, "srcParticlePos");
        bind_vec3(
            &program,
            &mut bindings,
            &mut out_bindings,
            &srcParticlePos.borrow(),
            "srcParticlePos".to_string(),
        );

        const BIND_CONTEXT_9: [&str; 32] = update_bind_context!(BIND_CONTEXT_8, "srcParticleVel");
        bind_vec3(
            &program,
            &mut bindings,
            &mut out_bindings,
            &srcParticleVel.borrow(),
            "srcParticleVel".to_string(),
        );

        const BIND_CONTEXT_10: [&str; 32] = update_bind_context!(BIND_CONTEXT_9, "trianglePos");
        bind_vec3(
            &program,
            &mut bindings,
            &mut out_bindings,
            &triangle,
            "trianglePos".to_string(),
        );

        ready_to_run(BIND_CONTEXT_10);
        let result = run(program, &mut bindings, out_bindings);

        // Need to finagle async too allow reading here
        let dstParticlePos = read_fvec3(program, &result, "dstParticlePos").await;
        let dstParticleVel = read_fvec3(program, &result, "dstParticleVel").await;
        let trianglePos = read_fvec3(program, &result, "trianglePos").await;
        println!("Current values");
        println!("{:?}", dstParticlePos);
        println!("{:?}", dstParticleVel);
        static_assertions::const_assert!(can_pipe(&BIND_CONTEXT_10, &NEXT_STARTING_CONTEXT));
        graphics_pipe(
            &graphics_program,
            graphics_program
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
            graphics_bindings,
            graphics_out_bindings,
            frame,
            result,
        );

        let mut dstParticlePos1: [f32; 3] = Default::default();
        let mut dstParticlePos2: [f32; 3] = Default::default();
        let mut dstParticleVel1: [f32; 3] = Default::default();
        let mut dstParticleVel2: [f32; 3] = Default::default();

        dstParticlePos1.copy_from_slice(&dstParticlePos[0..3]);
        dstParticlePos2.copy_from_slice(&dstParticlePos[3..6]);
        dstParticleVel1.copy_from_slice(&dstParticleVel[0..3]);
        dstParticleVel2.copy_from_slice(&dstParticleVel[3..6]);

        srcParticlePos.replace(vec![dstParticlePos1, dstParticlePos2]);
        srcParticleVel.replace(vec![dstParticleVel1, dstParticleVel2]);
    };

    event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Everything that can be processed has been so we can now redraw the image on our window
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                futures::executor::block_on(draw(
                    &mut swap_chain.borrow_mut(),
                    &program,
                    &graphics_program,
                    &template_bindings,
                    &template_out_bindings,
                    &template_graphics_bindings,
                    &template_graphics_out_bindings,
                    &mut srcParticlePos,
                    &mut srcParticleVel,
                ));
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            // Ignore any other types of events
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    execute_gpu(event_loop, window);
}
