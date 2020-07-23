#[macro_use]
extern crate pipeline;

// use for the shader! macro
pub use pipeline::wgpu_compute_header;

pub use pipeline::wgpu_compute_header::{
    bind_vec, can_pipe, compile, new_bind_scope, pipe, read_vec, ready_to_run, run, SHADER,
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

    const ADD_ONE: (SHADER, [&str; 32], [&str; 32]) = shader! {
        [[buffer loop in] uint[]] add_one_in;
        [[buffer out] uint[]] add_two_in;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_in[index] = add_one_in[index]+1;
            }
        }}
    };

    const ADD_TWO: (SHADER, [&str; 32], [&str; 32]) = shader! {
        [[buffer loop in] uint[]] add_two_in;
        [[buffer out] uint[]] add_two_result;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_result[index] = add_two_in[index]+2;
            }
        }}
    };

    const S1: SHADER = ADD_ONE.0;
    const STARTING_BIND_CONTEXT: [&str; 32] = ADD_ONE.1;
    const ENDING_BIND_CONTEXT: [&str; 32] = ADD_ONE.2;
    let (program1, mut bindings1, mut out_bindings1) = compile(&S1).await;

    const S2: SHADER = ADD_TWO.0;
    const NEXT_STARTING_CONTEXT: [&str; 32] = ADD_TWO.1;
    let (program2, bindings2, out_bindings2) = compile(&S2).await;

    let indices: Vec<u32> = vec![1, 2, 3, 4];

    const BIND_CONTEXT_1: [&str; 32] = update_bind_context!(STARTING_BIND_CONTEXT, "add_one_in");
    bind_vec(
        &program1,
        &mut bindings1,
        &mut out_bindings1,
        &indices,
        "add_one_in".to_string(),
    );
    {
        ready_to_run(BIND_CONTEXT_1);
        let result = run(&program1, &mut bindings1, out_bindings1);
        println!("{:?}", read_vec(&program1, &result, "add_two_in").await);
        static_assertions::const_assert!(can_pipe(&ENDING_BIND_CONTEXT, &NEXT_STARTING_CONTEXT));
        let pipe_result = pipe(&program2, bindings2, out_bindings2, result);
/*         println!("{:?}", read_vec(&program2, &pipe_result, "add_two_in").await); */
        println!("{:?}", read_vec(&program2, &pipe_result, "add_two_result").await);
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
