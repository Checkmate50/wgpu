mod wgpu_compute_header;

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
    let s = shader! {
        [[buffer loop in out] uint[]] indices;
        [[buffer in] uint[]] indices2;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;

        void main() {
            // uint xindex = gl_GlobalInvocationID.x;
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    };

    let (program, mut bindings, mut out_bindings) = wgpu_compute_header::compile(&s).await;
    let (_, _, mut out_bindings2) = wgpu_compute_header::compile(&s).await;

    let indices = vec![1, 2, 3, 4];
    let indices2_1 = vec![1, 2, 3, 4];
    let indices2_2 = vec![4, 3, 2, 1];

    wgpu_compute_header::bind_vec(
        &program,
        &mut bindings,
        &mut out_bindings,
        &indices,
        "indices".to_string(),
    );
    {
        wgpu_compute_header::bind_vec(
            &program,
            &mut bindings,
            &mut out_bindings,
            &indices2_1,
            "indices2".to_string(),
        );
        {
            // Todo have some write or result function that captures/uses the result instead of returning it
            println!(
                "{:?}",
                wgpu_compute_header::run(&program, &mut bindings, out_bindings).await
            );
        }
    }

    wgpu_compute_header::bind_vec(
        &program,
        &mut bindings,
        &mut out_bindings2,
        &indices,
        "indices".to_string(),
    );
    {
        /*     wgpu_compute_header::bind_vec(&program, &mut bindings, &indices, "indices".to_string()); */
        wgpu_compute_header::bind_vec(
            &program,
            &mut bindings,
            &mut out_bindings2,
            &indices2_2,
            "indices2".to_string(),
        );
        {
            println!(
                "{:?}",
                wgpu_compute_header::run(&program, &mut bindings, out_bindings2).await
            );
        }
    }
}

fn main() {
    env_logger::init();

    futures::executor::block_on(execute_gpu());
}
