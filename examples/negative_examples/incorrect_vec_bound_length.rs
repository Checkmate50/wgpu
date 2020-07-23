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
    bind_float, bind_vec, bind_vec2, compile, new_bind_scope, read_fvec, ready_to_run, run, SHADER,
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

    // TODO how to set a work group larger than 1?
    const BOIDS: (SHADER, [&str; 32], [&str; 32]) = shader! {
        [[uniform in] float] deltaT;
        [[uniform in] float] rule1Distance;
        [[uniform in] float] rule2Distance;
        [[uniform in] float] rule3Distance;
        [[uniform in] float] rule1Scale;
        [[uniform in] float] rule2Scale;
        [[uniform in] float] rule3Scale;

        [[buffer loop in] vec2[]] srcParticlePos;
        [[buffer loop in] vec2[]] srcParticleVel;
        [[buffer out] vec2[]] dstParticlePos;
        [[buffer out] vec2[]] dstParticleVel;


    /* layout(std140, set = 0, binding = 1) buffer SrcParticles {
        Particle particles[NUM_PARTICLES];
    } srcParticles;
    layout(std140, set = 0, binding = 2) buffer DstParticles {
        Particle particles[NUM_PARTICLES];
    } dstParticles; */

            {{
    // TODO This would be nice
    /* struct Particle {
        vec2 pos;
        vec2 vel;
    }; */

    // TODO This is inserted in the wgpu example

    void main() {
        // https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
        uint index = gl_GlobalInvocationID.x;

    /*     if (index >= NUM_PARTICLES) { return; } */
        vec2 vPos = srcParticlePos[index];
        vec2 vVel = srcParticleVel[index];
        vec2 cMass = vec2(0.0, 0.0);
        vec2 cVel = vec2(0.0, 0.0);
        vec2 colVel = vec2(0.0, 0.0);
        int cMassCount = 0;
        int cVelCount = 0;
        vec2 pos;
        vec2 vel;
        // The iteration of the number of particles was set by a #define value
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
        dstParticlePos[index] = vPos;
        dstParticleVel[index] = vVel;
    }
            }}
        };

    let mut srcParticlePos: Vec<f32> = vec![0.0, 0.0, 0.3, 0.2, 0.5];
    let mut srcParticleVel: Vec<f32> = vec![0.01, -0.02, -0.05, -0.03];
    let deltaT: f32 = 0.04;
    let rule1Distance: f32 = 0.1;
    let rule2Distance: f32 = 0.25;
    let rule3Distance: f32 = 0.25;
    let rule1Scale: f32 = 0.02;
    let rule2Scale: f32 = 0.05;
    let rule3Scale: f32 = 0.005;
    while true {
        const S: SHADER = BOIDS.0;
        const STARTING_BIND_CONTEXT: [&str; 32] = BOIDS.1;

        let (program, mut bindings, mut out_bindings) = compile(&S).await;

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
        bind_vec2(
            &program,
            &mut bindings,
            &mut out_bindings,
            &srcParticlePos,
            "srcParticlePos".to_string(),
        );

        const BIND_CONTEXT_9: [&str; 32] = update_bind_context!(BIND_CONTEXT_8, "srcParticleVel");
        bind_vec2(
            &program,
            &mut bindings,
            &mut out_bindings,
            &srcParticleVel,
            "srcParticleVel".to_string(),
        );

        {
            ready_to_run(BIND_CONTEXT_9);
            let result = run(&program, &mut bindings, out_bindings);
            let dstParticlePos = read_fvec(&program, &result, "dstParticlePos").await;
            let dstParticleVel = read_fvec(&program, &result, "dstParticleVel").await;
            println!("Current values");
            println!("{:?}", dstParticlePos);
            println!("{:?}", dstParticleVel);
            srcParticlePos = dstParticlePos;
            srcParticleVel = dstParticleVel;
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
