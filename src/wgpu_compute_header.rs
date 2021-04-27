use glsl_to_spirv::ShaderType;

use std::collections::HashMap;

use crate::shared::{check_gl_builtin_type, compile_shader, process_body, PARAMETER, QUALIFIER};

use crate::bind::DefaultBinding;

pub struct ComputeProgram {
    pub pipeline: wgpu::ComputePipeline,
}

#[derive(Debug)]
pub struct ComputeBindings {
    pub bindings: Vec<DefaultBinding>,
}

#[derive(Debug)]
pub struct OutComputeBindings {
    pub bindings: Vec<DefaultBinding>,
}

//todo unify this with the graphics version into shared
fn stringify_shader(s: &ComputeShader, b: &ComputeBindings, b_out: &OutComputeBindings) -> String {
    let mut buffer = Vec::new();
    for i in &b.bindings[..] {
        buffer.push(format!(
            "layout(set = {}, binding = {}) {} BINDINGS{}{} {{\n",
            i.group_number,
            i.binding_number,
            if i.qual.contains(&QUALIFIER::BUFFER) {
                "buffer"
            } else if i.qual.contains(&QUALIFIER::UNIFORM) {
                "uniform"
            } else {
                panic!(
                    "You are trying to do something with something that isn't a buffer or uniform"
                )
            },
            i.group_number,
            i.binding_number
        ));

        buffer.push(i.gtype.to_string() + " " + &i.name + ";\n");
        buffer.push("};\n".to_string());
    }
    for i in &b_out.bindings[..] {
        if i.qual.contains(&QUALIFIER::BUFFER) {
            buffer.push(format!(
                "layout(set = {}, binding = {}) buffer BINDINGS{}{} {{\n",
                i.group_number, i.binding_number, i.group_number, i.binding_number
            ));

            buffer.push(i.gtype.to_string() + " " + &i.name + ";\n");
            buffer.push("};\n".to_string());
        }
    }
    format!(
        //todo figure out how to use a non-1 local size
        "#version 450\nlayout(local_size_x = 1) in;\n{}\n\n{}\n",
        buffer.join(""),
        process_body(s.body)
    )
}

fn create_bindings(compute: &ComputeShader) -> (ComputeBindings, OutComputeBindings) {
    let mut binding_struct = Vec::new();
    let mut out_binding_struct = Vec::new();
    let mut group_map = HashMap::new();
    let mut uniform_binding_number_map = HashMap::new();
    let mut group_set_number = 0;
    for i in &compute.params[..] {
        // Bindings that are kept between runs
        if !check_gl_builtin_type(i.name, &i.gtype) {
            if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                let group_number = match group_map.get(i.group.unwrap()) {
                    Some(i) => *i,
                    None => {
                        let x = group_set_number;
                        group_map.insert(i.group.unwrap(), x);
                        group_set_number += 1;
                        x
                    }
                };
                let uniform_binding_number =
                    uniform_binding_number_map.entry(group_number).or_insert(0);
                binding_struct.push(DefaultBinding {
                    binding_number: *uniform_binding_number,
                    group_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                *uniform_binding_number += 1;
            // Bindings that are invalidated after a run
            } else if i.qual.contains(&QUALIFIER::OUT) {
                let group_number = match group_map.get(i.group.unwrap()) {
                    Some(i) => *i,
                    None => {
                        let x = group_set_number;
                        group_map.insert(i.group.unwrap(), x);
                        group_set_number += 1;
                        x
                    }
                };
                let uniform_binding_number =
                    uniform_binding_number_map.entry(group_number).or_insert(0);
                out_binding_struct.push(DefaultBinding {
                    binding_number: *uniform_binding_number,
                    group_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                *uniform_binding_number += 1;
            }
        }
    }

    (
        ComputeBindings {
            bindings: binding_struct,
        },
        OutComputeBindings {
            bindings: out_binding_struct,
        },
    )
}

pub async fn compile(
    compute: &ComputeShader,
    device: &wgpu::Device,
    bind_group_layout: Vec<wgpu::BindGroupLayout>,
) -> ComputeProgram {
    let (program_bindings, out_program_bindings) = create_bindings(&compute);

    let cs_module = compile_shader(
        stringify_shader(&compute, &program_bindings, &out_program_bindings),
        ShaderType::Compute,
        &device,
    );

    let bind_group_layout_ref: Vec<&wgpu::BindGroupLayout> =
        bind_group_layout.iter().map(|a| a).collect();

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bind_group_layout_ref,
        push_constant_ranges: &[],
    });

    // The part where we actually bring it all together
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    ComputeProgram { pipeline }
}

pub fn compute_run(mut rpass: wgpu::ComputePass, length: u32) -> wgpu::ComputePass {
    {
        compute(&mut rpass, length);
    }
    rpass
}

pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
    cpass.dispatch(length, 1, 1);
}

#[derive(Debug)]
pub struct ComputeShader {
    pub params: &'static [PARAMETER],
    pub body: &'static str,
}

#[macro_export]
macro_rules! compute_shader {
        ($($body:tt)*) => {{
            const S : (&[pipeline::shared::PARAMETER], &'static str) = shader!($($body)*);
            (pipeline::wgpu_compute_header::ComputeShader{params:S.0, body:S.1})
        }};
    }
