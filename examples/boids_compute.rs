#![recursion_limit = "1024"]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

pub use pipeline::wgpu_compute_header::{compile, compute_run, ComputeShader};

pub use pipeline::bind::{BindGroup1, BindGroup2, BindGroup3, BufferData, Indices, Vertex};
pub use pipeline::AbstractBind;

use std::convert::TryInto;
use std::rc::Rc;

async fn execute_gpu() {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropiate adapter");

    // The device manages the connection and resources of the adapter
    // The queue is a literal queue of tasks for the gpu
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits {
                    max_bind_groups: 5,
                    ..Default::default()
                },
            },
            None,
        )
        .await
        .expect("Failed to create device");

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
    my_shader! {BOIDS = {
        [group1 [uniform in] float] deltaT;
        [group2 [uniform in] float] rule1Distance;
        [group2 [uniform in] float] rule2Distance;
        [group2 [uniform in] float] rule3Distance;
        [group3 [uniform in] float] rule1Scale;
        [group3 [uniform in] float] rule2Scale;
        [group3 [uniform in] float] rule3Scale;

        [group4 [buffer loop in] vec3[]] srcParticlePos;
        [group4 [buffer loop in] vec3[]] srcParticleVel;
        [group5 [buffer in out] vec3[]] dstParticlePos;
        [group5 [buffer in out] vec3[]] dstParticleVel;


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
    }}

    const S: ComputeShader = eager_compute_shader! {BOIDS!()};
    eager_binding! {context = BOIDS!()};

    let program = compile(&S, &device, context.get_layout(&device)).await;

    let srcParticlePos = BufferData::new(vec![[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]]);
    let srcParticleVel = BufferData::new(vec![[0.01, -0.02, 0.0], [-0.05, -0.03, 0.0]]);
    let srcParticle_bg = BindGroup2::new(&device, &srcParticlePos, &srcParticleVel);

    let dstParticlePos = BufferData::new(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let dstParticleVel = BufferData::new(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let dstParticle_bg = BindGroup2::new(&device, &dstParticlePos, &dstParticleVel);

    let deltaT = BindGroup1::new(&device, &BufferData::new(0.04));

    let ruleDistance = BindGroup3::new(
        &device,
        &BufferData::new(0.1),
        &BufferData::new(0.25),
        &BufferData::new(0.25),
    );

    let ruleScale = BindGroup3::new(
        &device,
        &BufferData::new(0.02),
        &BufferData::new(0.05),
        &BufferData::new(0.005),
    );

    let mut loop_count = 0;
    loop {
        let (srcParticle, dstParticle) = if loop_count % 2 == 0 {
            (&srcParticle_bg, &dstParticle_bg)
        } else {
            (&dstParticle_bg, &srcParticle_bg)
        };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&program.pipeline);

            let context1 = (&context).set_deltaT(&mut cpass, &deltaT);

            let context2 =
                context1.set_rule1Distance_rule2Distance_rule3Distance(&mut cpass, &ruleDistance);

            let context3 = context2.set_rule1Scale_rule2Scale_rule3Scale(&mut cpass, &ruleScale);

            let context4 = context3.set_srcParticlePos_srcParticleVel(&mut cpass, &srcParticle);

            let context5 = context4.set_dstParticlePos_dstParticleVel(&mut cpass, &dstParticle);

            {
                let _ = context5.runnable(|| compute_run(cpass, 2));
            }
        }
        let dstParticlePos = dstParticle.setup_read_0(
            &device,
            &mut encoder,
            0..std::mem::size_of::<f32>() as u64 * 8,
        );
        let dstParticleVel = dstParticle.setup_read_1(
            &device,
            &mut encoder,
            0..std::mem::size_of::<f32>() as u64 * 8,
        );

        queue.submit(Some(encoder.finish()));

        println!(
            "{:?}",
            dstParticlePos
                .read(&device)
                .await
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>()
                .as_slice()
                .chunks_exact(4)
                .map(|v4| TryInto::<[f32; 4]>::try_into(v4).unwrap())
                .map(|v4| [v4[0], v4[1], v4[2]])
                .collect::<Vec<[f32; 3]>>()
        );
        println!(
            "{:?}",
            dstParticleVel
                .read(&device)
                .await
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>()
                .as_slice()
                .chunks_exact(4)
                .map(|v4| TryInto::<[f32; 4]>::try_into(v4).unwrap())
                .map(|v4| [v4[0], v4[1], v4[2]])
                .collect::<Vec<[f32; 3]>>()
        );

        loop_count += 1;
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
