/* #version 450

// These should match the Rust constants defined in main.rs
#define NUM_PARTICLES 1500
#define PARTICLES_PER_GROUP 64

layout(local_size_x = PARTICLES_PER_GROUP) in;

struct Particle {
    vec2 pos;
    vec2 vel;
};
layout(std140, set = 0, binding = 0) uniform SimParams {
    float deltaT;
    float rule1Distance;
    float rule2Distance;
    float rule3Distance;
    float rule1Scale;
    float rule2Scale;
    float rule3Scale;
} params;
layout(std140, set = 0, binding = 1) buffer SrcParticles {
    Particle particles[NUM_PARTICLES];
} srcParticles;
layout(std140, set = 0, binding = 2) buffer DstParticles {
    Particle particles[NUM_PARTICLES];
} dstParticles;

void main() {
    // https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
    uint index = gl_GlobalInvocationID.x;

    if (index >= NUM_PARTICLES) { return; }
    vec2 vPos = srcParticles.particles[index].pos;
    vec2 vVel = srcParticles.particles[index].vel;
    vec2 cMass = vec2(0.0, 0.0);
    vec2 cVel = vec2(0.0, 0.0);
    vec2 colVel = vec2(0.0, 0.0);
    int cMassCount = 0;
    int cVelCount = 0;
    vec2 pos;
    vec2 vel;
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        if (i == index) { continue; }
        pos = srcParticles.particles[i].pos.xy;
        vel = srcParticles.particles[i].vel.xy;
        if (distance(pos, vPos) < params.rule1Distance) {
            cMass += pos;
            cMassCount++;
        }
        if (distance(pos, vPos) < params.rule2Distance) {
            colVel -= (pos - vPos);
        }
        if (distance(pos, vPos) < params.rule3Distance) {
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
    vVel += cMass * params.rule1Scale + colVel * params.rule2Scale + cVel * params.rule3Scale;
    // clamp velocity for a more pleasing simulation.
    vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
    // kinematic update
    vPos += vVel * params.deltaT;
    // Wrap around boundary
    if (vPos.x < -1.0) vPos.x = 1.0;
    if (vPos.x > 1.0) vPos.x = -1.0;
    if (vPos.y < -1.0) vPos.y = 1.0;
    if (vPos.y > 1.0) vPos.y = -1.0;
    dstParticles.particles[index].pos = vPos;
    // Write back
    dstParticles.particles[index].vel = vVel;
} */

#[macro_use]
extern crate pipeline;

// use for the shader! macro
pub use pipeline::wgpu_compute_header;

pub use pipeline::wgpu_compute_header::{
    bind_vec, compile, new_bind_scope, ready_to_run, run, SHADER,
};
pub use static_assertions::const_assert;

async fn execute_gpu() {
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

    // todo take the comment above and put it here
    const TRIVIAL: (SHADER, [&str; 32], [&str; 32]) = shader! {
            [[buffer loop in out] uint[]] indices;
            //[[buffer out] uint[]] result;
            //[... uint] xindex;
            {{
    uint collatz_iterations(uint n) {
        uint i = 0;
        while(n != 1) {
            if (mod(n, 2) == 0) {
                n = n / 2;
            }
            else {
                n = (3 * n) + 1;
            }
            i++;
        }
        return i;
    }

    void main() {
        uint index = gl_GlobalInvocationID.x;
        indices[index] = collatz_iterations(indices[index]);
    }
            }}
        };

    const S: SHADER = TRIVIAL.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = TRIVIAL.1;

    let (program, mut bindings, mut out_bindings) = compile(&S).await;
    let (_, _, mut out_bindings2) = compile(&S).await;

    let indices: Vec<u32> = vec![1, 2, 3, 4];

    /*     const BIND_CONTEXT_1: ([&str; 32], bool) = new_bind_scope(&STARTING_BIND_CONTEXT, "indices");
    const_assert!(BIND_CONTEXT_1.1); */

    macro_rules! update_bind_context {
        ($bind_context:tt, $bind_name:tt) => {{
            const BIND_CONTEXT: ([&str; 32], bool) = new_bind_scope(&$bind_context, $bind_name);
            const_assert!(BIND_CONTEXT.1);
            BIND_CONTEXT.0
        }};
    }

    const BIND_CONTEXT_1: [&str; 32] = update_bind_context!(STARTING_BIND_CONTEXT, "indices");
    bind_vec(
        &program,
        &mut bindings,
        &mut out_bindings,
        &indices,
        "indices".to_string(),
    );

    {
        // Todo have some write or result function that captures/uses the result instead of returning it
        ready_to_run(BIND_CONTEXT_1);
        println!("{:?}", run(&program, &mut bindings, out_bindings).await);
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
