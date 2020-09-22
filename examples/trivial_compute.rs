#![feature(const_panic)]

#[macro_use]
extern crate pipeline;

pub use pipeline::wgpu_compute_header::{compile, read_uvec, run, ComputeShader};

pub use pipeline::shared::{is_gl_builtin, Bindable, Context};

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

    struct ShaderContext_indices_indices2 {}
    struct ShaderContext_indices2 {}
    struct ShaderContext_indices {}
    struct ShaderContext {}

    impl ShaderContext_indices_indices2 {
        pub fn bind_indices(&self) -> ShaderContext_indices2 {
            ShaderContext_indices2 {}
        }
        pub fn bind_indices2(&self) -> ShaderContext_indices {
            ShaderContext_indices {}
        }
    }

    impl ShaderContext_indices2 {
        pub fn bind_indices2(self) -> ShaderContext {
            ShaderContext {}
        }
    }

    impl ShaderContext_indices {
        pub fn bind_indices(&self) -> ShaderContext {
            ShaderContext {}
        }
    }

    impl ShaderContext {
        pub fn run(self) {}
        // Do compute shader context's have a pipe function?
    }

    const S: ComputeShader = TRIVIAL.0;
    const STARTING_BIND_CONTEXT: BindingContext = TRIVIAL.1;

    let (program, mut bindings, mut out_bindings) = compile(&S).await;

    let indices_1: Vec<u32> = vec![1, 2, 3, 4];
    let indices_2: Vec<u32> = vec![2, 2, 2, 2];
    let indices2: Vec<u32> = vec![4, 3, 2, 1];

    let context = Context::new();
    {
        const BIND_CONTEXT_1: BindingContext =
            update_bind_context(&STARTING_BIND_CONTEXT, "indices2");
        let context1 = bind!(
            program,
            bindings,
            out_bindings,
            "indices2",
            indices2,
            context,
            BIND_CONTEXT_1
        );
        {
            const BIND_CONTEXT_2: BindingContext = update_bind_context(&BIND_CONTEXT_1, "indices");
            let _ = bind_mutate!(
                program,
                bindings,
                out_bindings,
                "indices",
                indices_1,
                context1,
                BIND_CONTEXT_2
            );
            {
                const _: () = ready_to_run(BIND_CONTEXT_2);
                let result_out_bindings = out_bindings.move_buffers();
                let result1 = run(&program, &mut bindings, result_out_bindings);
                println!("{:?}", read_uvec(&program, &result1, "indices").await);
            }
        }
        {
            const BIND_CONTEXT_4: BindingContext = update_bind_context(&BIND_CONTEXT_1, "indices");
            let _ = bind_mutate!(
                program,
                bindings,
                out_bindings,
                "indices",
                indices_2,
                context1,
                BIND_CONTEXT_4
            );
            {
                const _: () = ready_to_run(BIND_CONTEXT_4);
                let result1 = run(&program, &mut bindings, out_bindings);
                println!("{:?}", read_uvec(&program, &result1, "indices").await);
            }
        }
    }
}

fn main() {
    futures::executor::block_on(execute_gpu());
}
