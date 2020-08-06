#[macro_use]
extern crate pipeline;

// use for the shader! macro
pub use pipeline::shared;
pub use pipeline::wgpu_compute_header;

pub use pipeline::shared::{bind_vec, is_gl_builtin, new_bind_scope, ready_to_run};
pub use pipeline::wgpu_compute_header::{compile, read_uvec, run, ComputeShader};

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

    const TRIVIAL: (ComputeShader, [&str; 32], [&str; 32]) = compute_shader! {
        [[buffer loop in out] uint[]] indices;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            uint collatz_iterations(uint n) {
                uint i = 0;
                while(n > 1) {
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

    const S: ComputeShader = TRIVIAL.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = TRIVIAL.1;

    let (program, mut bindings, mut out_bindings) = compile(&S).await;

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
        ready_to_run(BIND_CONTEXT_1);
        let result = run(&program, &mut bindings, out_bindings);
        println!("{:?}", read_uvec(&program, &result, "indices").await);
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
