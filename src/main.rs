mod wgpu_compute_header;

async fn execute_gpu() {
    // todo needs out and loop annotations
    // values currently hardcoded
    let s = shader! {
        [buffer uint[]] indices;
        [buffer uint[]] indices2;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    };

    let (program, bindings) = wgpu_compute_header::compile(s).await;

    let new_bindings =
        wgpu_compute_header::bind_vec(&bindings, &vec![1, 2, 3, 4], "indices".to_string());
    {
        let final_bindings =
            wgpu_compute_header::bind_vec(&new_bindings, &vec![1, 2, 3, 4], "indices2".to_string());
        // Todo have some write or result function that captures/uses the result instead of returning it
        println!("{:?}", wgpu_compute_header::run(&program, final_bindings).await);
    }

    let new_bindings =
        wgpu_compute_header::bind_vec(&bindings, &vec![1, 2, 3, 4], "indices".to_string());
    let final_bindings2 =
        wgpu_compute_header::bind_vec(&new_bindings, &vec![4, 3, 2, 1], "indices2".to_string());
    println!("{:?}", wgpu_compute_header::run(&program, final_bindings2).await);
}

fn main() {
    env_logger::init();

    futures::executor::block_on(execute_gpu());
}
