#![feature(const_panic)]

#[macro_use]
extern crate pipeline;

pub use pipeline::shared::{is_gl_builtin, Bindable, Context};
pub use pipeline::wgpu_compute_header::{compile, read_uvec, run, ComputeShader};

pub use pipeline::context::{ready_to_run, update_bind_context, BindingContext};

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

    const TRIVIAL: (ComputeShader, BindingContext) = compute_shader! {
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
    const STARTING_BIND_CONTEXT: BindingContext = TRIVIAL.1;

    let (program, mut bindings, mut out_bindings) = compile(&S).await;

    let indices: Vec<u32> = vec![1, 2, 3, 4];

    let context = Context::new();
    {
        const BIND_CONTEXT_1: BindingContext =
            update_bind_context(&STARTING_BIND_CONTEXT, "indices");
        let context1 = bind_mutate!(
            program,
            bindings,
            out_bindings,
            "indices",
            indices,
            context,
            BIND_CONTEXT_1
        );
        {
            const _: () = ready_to_run(BIND_CONTEXT_1);
            let result = run(&program, &mut bindings, out_bindings);
            println!("{:?}", read_uvec(&program, &result, "indices").await);
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
