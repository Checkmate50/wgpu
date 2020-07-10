mod wgpu_compute_header;

async fn execute_gpu() {
    // todo needs out and loop annotations
    // values currently hardcoded
    // in and out don't do anything yet
    let s = shader! {
        [[buffer in out] uint[]] indices;
        [[buffer in] uint[]] indices2;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    };

    let (program, mut bindings) = wgpu_compute_header::compile(s).await;

    let indices = vec![1, 2, 3, 4];
    let indices2_1 = vec![1, 2, 3, 4];
    let indices2_2 = vec![4, 3, 2, 1];

    wgpu_compute_header::bind_vec(&program, &mut bindings, &indices, "indices".to_string());
    {
        wgpu_compute_header::bind_vec(&program, &mut bindings, &indices2_1, "indices2".to_string());
        {
            // Todo have some write or result function that captures/uses the result instead of returning it
            println!(
                "{:?}",
                wgpu_compute_header::run(&program, &mut bindings).await
            );
        }
    }

    wgpu_compute_header::bind_vec(&program, &mut bindings, &indices, "indices".to_string());
    {
        /*     wgpu_compute_header::bind_vec(&program, &mut bindings, &indices, "indices".to_string()); */
        wgpu_compute_header::bind_vec(&program, &mut bindings, &indices2_2, "indices2".to_string());
        {
            println!(
                "{:?}",
                wgpu_compute_header::run(&program, &mut bindings).await
            );
        }
    }
}

fn main() {
    env_logger::init();

    futures::executor::block_on(execute_gpu());
}
