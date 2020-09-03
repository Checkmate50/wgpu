#[macro_use]
extern crate pipeline;

// use for the shader! macro
pub use pipeline::shared;
pub use pipeline::wgpu_compute_header;

pub use pipeline::wgpu_compute_header::{compile, read_uvec, run, ComputeShader};

pub use pipeline::shared::{bind_vec, is_gl_builtin};

pub use pipeline::context::{ready_to_run, update_bind_context, BindingContext, MetaContext};

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

    const TRIVIAL: (ComputeShader, BindingContext) = compute_shader! {
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
    };

    const S: ComputeShader = TRIVIAL.0;
    const STARTING_BIND_CONTEXT: BindingContext = TRIVIAL.1;
    const STARTING_META_CONTEXT: MetaContext = MetaContext::new();

    let (program, mut bindings, mut out_bindings) = compile(&S).await;
    let (_, _, mut out_bindings2) = compile(&S).await;

    let indices: Vec<u32> = vec![1, 2, 3, 4];
    let indices2_1: Vec<u32> = vec![1, 2, 3, 4];
    let indices2_2: Vec<u32> = vec![4, 3, 2, 1];

    #[allow(dead_code)]
    {
        const BIND_CONTEXT_1: (BindingContext, MetaContext) = update_bind_context(
            &STARTING_BIND_CONTEXT,
            "indices",
            STARTING_META_CONTEXT,
            "BIND_CONTEXT_1",
        );
        {
            const BIND_CONTEXT_2: (BindingContext, MetaContext) = update_bind_context(
                &BIND_CONTEXT_1.0,
                "indices2",
                BIND_CONTEXT_1.1,
                "BIND_CONTEXT_2",
            );

            const NEXT_META_CONTEXT1: MetaContext = {
                const NEXT_META_CONTEXT: MetaContext =
                    ready_to_run(BIND_CONTEXT_2.0, BIND_CONTEXT_2.1);
                NEXT_META_CONTEXT
            };
            const BIND_CONTEXT_4: (BindingContext, MetaContext) = update_bind_context(
                &BIND_CONTEXT_1.0,
                "indices2",
                NEXT_META_CONTEXT1,
                "BIND_CONTEXT_4",
            );
            {
                const FINAL_META_CONTEXT: MetaContext =
                    ready_to_run(BIND_CONTEXT_4.0, BIND_CONTEXT_4.1);
            }
        }
    }

    bind_vec(
        &program,
        &mut bindings,
        &mut out_bindings,
        &indices,
        "indices".to_string(),
    );
    {
        bind_vec(
            &program,
            &mut bindings,
            &mut out_bindings,
            &indices2_1,
            "indices2".to_string(),
        );
        {
            let result1 = run(&program, &mut bindings, out_bindings);
            println!("{:?}", read_uvec(&program, &result1, "indices").await);
        }
        bind_vec(
            &program,
            &mut bindings,
            &mut out_bindings2,
            &indices2_2,
            "indices2".to_string(),
        );
        {

            let result1 = run(&program, &mut bindings, out_bindings2);
            println!("{:?}", read_uvec(&program, &result1, "indices").await);
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
