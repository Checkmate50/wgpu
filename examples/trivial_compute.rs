#![recursion_limit = "256"]
#[macro_use]
extern crate pipeline;

#[macro_use]
extern crate eager;

pub use pipeline::wgpu_compute_header::{compile, read_uvec, run, ComputeShader};

pub use wgpu_macros::{generic_bindings, init};

async fn execute_gpu() {
    init!();

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

    my_shader! {compute = {
        [[buffer loop in out] uint[]] indices;
        [[buffer in] uint[]] indices2;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                indices[index] = indices[index]+indices2[index];
            }
        }}
    }}

    const S: ComputeShader = eager_compute_shader!{compute!()};

    eager_binding!{context = compute!()};

    let (program, mut bindings, mut out_bindings) = compile(&S).await;

    let indices_1: Vec<u32> = vec![1, 2, 3, 4];
    let indices_2: Vec<u32> = vec![2, 2, 2, 2];
    let indices2: Vec<u32> = vec![4, 3, 2, 1];

    {

        let context1 = context.bind_indices2(&indices2, &program, &mut bindings, &mut out_bindings);
        let context2 = (&context1).bind_indices(&indices_1, &program, &mut bindings, &mut out_bindings);
        let result_out_bindings = out_bindings.move_buffers();

        let result1 =
            context2.runable(|| run(&program, &mut bindings, result_out_bindings));

        println!("{:?}", read_uvec(&program, &result1, "indices").await);


        {
            let context3 =
                context1.bind_indices(&indices_2, &program, &mut bindings, &mut out_bindings);
            {
                let result1 = context3.runable(|| run(&program, &mut bindings, out_bindings));
                println!("{:?}", read_uvec(&program, &result1, "indices").await);
            }
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
