#![recursion_limit = "256"]

#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

pub use pipeline::wgpu_compute_header::{compile, pipe, read_uvec, run, ComputeShader};

pub use wgpu_macros::{generic_bindings, init};

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

    init!();

    my_shader! {One = {
        [[buffer loop in] uint[]] add_one_in;
        [[buffer out] uint[]] add_two_in;
        //unit ->
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_in[index] = add_one_in[index]+1;
            }
        }}
    }}

    const ADD_ONE: ComputeShader = eager! { lazy! {compute_shader! { eager!{One!()}}}};

    eager! { lazy! { generic_bindings! { context = eager!{ One!()}}}};

    //generic_bindings! {context = add_one_in; add_two_in}

    my_shader! {Two = {
        [[buffer loop in] uint[]] add_two_in;
        [[buffer out] uint[]] add_two_result;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                add_two_result[index] = add_two_in[index]+2;
            }
        }}
    }}
    const ADD_TWO: ComputeShader = eager! { lazy! {compute_shader! { eager!{Two!()}}}};
    eager! { lazy! { generic_bindings! { next_context = eager!{ Two!()}}}};
    //generic_bindings! { = add_two_in; add_two_result}

    let (program1, mut bindings1, mut out_bindings1) = compile(&ADD_ONE).await;

    let (program2, bindings2, out_bindings2) = compile(&ADD_TWO).await;

    let indices: Vec<u32> = vec![1, 2, 3, 4];

    {
        let context1 =
            context.bind_add_one_in(&indices, &program1, &mut bindings1, &mut out_bindings1);
        {
            let result =  context1.runable(|| run(&program1, &mut bindings1, out_bindings1));
            println!("{:?}", read_uvec(&program1, &result, "add_two_in").await);

            context1.can_pipe(&next_context);
            let pipe_result = pipe(&program2, bindings2, out_bindings2, result);
            /*         println!("{:?}", read_vec(&program2, &pipe_result, "add_two_in").await); */
            println!(
                "{:?}",
                read_uvec(&program2, &pipe_result, "add_two_result").await
            );
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
