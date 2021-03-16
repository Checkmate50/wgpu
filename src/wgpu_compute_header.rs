use glsl_to_spirv::ShaderType;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;

use crate::shared::{check_gl_builtin_type, compile_shader, process_body, PARAMETER, QUALIFIER};

use crate::bind::{
    new_bindings, Bindings, DefaultBinding, OutProgramBindings, ProgramBindings, SamplerBinding,
    TextureBinding,
};

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


//todo is this still needed??
impl OutComputeBindings {
    pub fn move_buffers(&mut self) -> OutComputeBindings {
        let mut new_binds = new_bindings(&self.bindings);
        for i in self.bindings.iter_mut() {
            let bind_pos = new_binds.iter_mut().position(|x| x.name == i.name).unwrap();

            new_binds[bind_pos].data = std::mem::replace(&mut i.data, None);
            new_binds[bind_pos].length = std::mem::replace(&mut i.length, None);
        }
        OutComputeBindings {
            bindings: new_binds,
        }
    }
}

impl Bindings for ComputeBindings {
    fn new(&self) -> ComputeBindings {
        ComputeBindings {
            bindings: new_bindings(&self.bindings),
        }
    }
}

impl ProgramBindings for ComputeBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding> {
        &mut self.bindings
    }
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding {
        &mut self.bindings[index]
    }
    fn get_samplers(&mut self) -> Option<&mut Vec<SamplerBinding>> {
        None
    }
    fn get_textures(&mut self) -> Option<&mut Vec<TextureBinding>> {
        None
    }
}

impl Bindings for OutComputeBindings {
    fn new(&self) -> OutComputeBindings {
        OutComputeBindings {
            bindings: new_bindings(&self.bindings),
        }
    }
}

impl OutProgramBindings for OutComputeBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding> {
        &mut self.bindings
    }
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding {
        &mut self.bindings[index]
    }
}

//todo unify this with the graphics version into shared
fn stringify_shader(s: &ComputeShader, b: &ComputeBindings, b_out: &OutComputeBindings) -> String {
    let mut buffer = Vec::new();
    for i in &b.bindings[..] {
        buffer.push(format!(
            "layout(binding = {}) {} BINDINGS{} {{\n",
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
            i.binding_number
        ));

        buffer.push(i.gtype.to_string() + " " + &i.name + ";\n");
        buffer.push("};\n".to_string());
    }
    for i in &b_out.bindings[..] {
        if i.qual.contains(&QUALIFIER::BUFFER) {
            buffer.push(format!(
                "layout(binding = {}) buffer BINDINGS{} {{\n",
                i.binding_number, i.binding_number
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

//todo remove bindgroup layout from this
fn create_bindings(
    compute: &ComputeShader,
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, ComputeBindings, OutComputeBindings) {
    let mut binding_struct: Vec<DefaultBinding> = Vec::new();
    let mut binding_number = 0;
    let mut out_binding_struct: Vec<DefaultBinding> = Vec::new();
    for i in &compute.params[..] {
        // Bindings that are kept between runs
        if !check_gl_builtin_type(i.name, &i.gtype) {
            if i.qual.contains(&QUALIFIER::IN) && !i.qual.contains(&QUALIFIER::OUT) {
                binding_struct.push(DefaultBinding {
                    binding_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                binding_number += 1;
            // Bindings that are invalidated after a run
            } else if i.qual.contains(&QUALIFIER::OUT) {
                out_binding_struct.push(DefaultBinding {
                    binding_number,
                    name: i.name.to_string(),
                    data: None,
                    length: None,
                    gtype: i.gtype.clone(),
                    qual: i.qual.to_vec(),
                });
                binding_number += 1;
            }
        }
    }

    // Create a layout for our bindings
    // If we had textures we would use this to lay them out

    let mut bind_entry = Vec::new();

    for i in &binding_struct {
        bind_entry.push(wgpu::BindGroupLayoutEntry {
            binding: i.binding_number,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: if i.qual.contains(&QUALIFIER::UNIFORM) {
                wgpu::BindingType::UniformBuffer { dynamic: false }
            } else {
                wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                }
            },
        });
    }

    for i in &out_binding_struct {
        bind_entry.push(wgpu::BindGroupLayoutEntry {
            binding: i.binding_number,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: if i.qual.contains(&QUALIFIER::UNIFORM) {
                wgpu::BindingType::UniformBuffer { dynamic: false }
            } else {
                wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                }
            },
        });
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &bind_entry,
        label: None,
    });

    (
        bind_group_layout,
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
    //todo bind_group_layout: Vec<wgpu::BindGroupLayout>,
) -> (ComputeProgram, ComputeBindings, OutComputeBindings) {

    let (bind_group_layout, program_bindings, out_program_bindings) =
        create_bindings(&compute, &device);

    let cs_module = compile_shader(
        stringify_shader(&compute, &program_bindings, &out_program_bindings),
        ShaderType::Compute,
        &device,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // The part where we actually bring it all together
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    (
        ComputeProgram {
            device,
            queue,
            pipeline,
            bind_group_layout,
        },
        program_bindings,
        out_program_bindings,
    )
}



fn buffer_map_setup<'a>(
    bindings: &'a ComputeBindings,
    out_bindings: &'a OutComputeBindings,
) -> HashMap<u32, &'a DefaultBinding> {
    let mut buffer_map = HashMap::new();

    for i in bindings.bindings.iter() {
        buffer_map.insert(i.binding_number, i);
    }

    for i in out_bindings.bindings.iter() {
        buffer_map.insert(i.binding_number, i);
    }
    buffer_map
}

pub fn run(
    program: &ComputeProgram,
    bindings: &ComputeBindings,
    mut out_bindings: OutComputeBindings,
) -> Vec<DefaultBinding> {
    let mut encoder = program
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Look for a loop qualifier in bindings, if it isn't there, it must be in out_bindings or we just give 1
    // Use this to get the size that the program should run over
    let mut bind = bindings
        .bindings
        .iter()
        .find(|i| i.qual.contains(&QUALIFIER::LOOP));

    if bind.is_none() {
        bind = out_bindings
            .bindings
            .iter()
            .find(|i| i.qual.contains(&QUALIFIER::LOOP));
    }

    let length = if let Some(b) = bind {
        b.length.unwrap()
    } else {
        1
    };

    for i in 0..(out_bindings.bindings.len()) {
        if !(out_bindings.bindings[i].qual.contains(&QUALIFIER::IN)) {
            out_bindings.bindings[i].data = Some(Rc::new(program.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: None,
                    size: (length * 3 * std::mem::size_of::<u32>() as u64) as u64,
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::MAP_READ
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC
                        | wgpu::BufferUsage::VERTEX,
                },
            )));
            out_bindings.bindings[i].length = Some(length);
        }
    }

    let buffer_map = buffer_map_setup(bindings, &out_bindings);

    let mut empty_vec = Vec::new();

    {
        for i in 0..(buffer_map.len()) {
            let b = buffer_map.get(&(i as u32)).expect(
                "I assumed all bindings would be buffers but I guess that has been invalidated",
            );
            empty_vec.push(wgpu::Binding {
                binding: b.binding_number,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &b
                        .data
                        .as_ref()
                        .unwrap_or_else(|| panic!("The binding of {} was not set", &b.name)),
                    range: 0..b
                        .length
                        .unwrap_or_else(|| panic!("The size of {} was not set", &b.name)),
                },
            });
        }

        let bgd = &wgpu::BindGroupDescriptor {
            layout: &program.bind_group_layout,
            bindings: empty_vec.as_slice(),
            label: None,
        };

        let bind_group = program.device.create_bind_group(bgd);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&program.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        compute(&mut cpass, length as u32);
    }
    program.queue.submit(&[encoder.finish()]);

    out_bindings.bindings
}

pub fn compute(cpass: &mut wgpu::ComputePass, length: u32) {
    cpass.dispatch(length, 1, 1);
}

pub async fn read_uvec(
    program: &ComputeProgram,
    results: &Vec<DefaultBinding>,
    name: &str,
) -> Vec<u32> {
    for i in results.iter() {
        if i.name == name {
            let result_buffer = i.data.as_ref().unwrap();
            // todo modify by size
            let size = i.length.unwrap() * std::mem::size_of::<u32>() as u64;
            let buffer_future = result_buffer.map_read(0, size);
            program.device.poll(wgpu::Maintain::Wait);

            if let Ok(mapping) = buffer_future.await {
                let x: Vec<u32> = mapping
                    .as_slice()
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                return x;
            } else {
                panic!("failed to run compute on gpu!");
            }
        }
    }
    panic!(
        "We didn't find the binding you were looking to read from: {}",
        name
    )
}

pub async fn read_fvec(
    program: &ComputeProgram,
    results: &Vec<DefaultBinding>,
    name: &str,
) -> Vec<f32> {
    for i in results.iter() {
        if i.name == name {
            let result_buffer = i.data.as_ref().unwrap();
            let size = i.length.unwrap() * std::mem::size_of::<f32>() as u64;
            let buffer_future = result_buffer.map_read(0, size);
            program.device.poll(wgpu::Maintain::Wait);

            if let Ok(mapping) = buffer_future.await {
                let x: Vec<f32> = mapping
                    .as_slice()
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                return x;
            } else {
                panic!("failed to run compute on gpu!");
            }
        }
    }
    panic!(
        "We didn't find the binding you were looking to read from: {}",
        name
    )
}

pub async fn read_fvec3(
    program: &ComputeProgram,
    results: &Vec<DefaultBinding>,
    name: &str,
) -> Vec<f32> {
    for i in results.iter() {
        if i.name == name {
            let result_buffer = i.data.as_ref().unwrap();
            let size = i.length.unwrap() * 3 * std::mem::size_of::<f32>() as u64;
            let buffer_future = result_buffer.map_read(0, size);
            program.device.poll(wgpu::Maintain::Wait);

            if let Ok(mapping) = buffer_future.await {
                let x: Vec<f32> = mapping
                    .as_slice()
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                return x;
            } else {
                panic!("failed to run compute on gpu!");
            }
        }
    }
    panic!(
        "We didn't find the binding you were looking to read from: {}",
        name
    )
}

pub fn pipe(
    program: &ComputeProgram,
    mut in_bindings: ComputeBindings,
    mut out_bindings: OutComputeBindings,
    result_vec: Vec<DefaultBinding>,
) -> Vec<DefaultBinding> {
    for i in result_vec {
        let binding = match in_bindings.bindings.iter().position(|x| x.name == i.name) {
            Some(x) => &mut in_bindings.bindings[x],
            None => {
                let x = out_bindings
                    .bindings
                    .iter()
                    .position(|x| x.name == i.name)
                    .expect("We couldn't find the binding");
                &mut out_bindings.bindings[x]
            }
        };

        /*          todo Check the types somewhere
        if !acceptable_types.contains(&binding.gtype) {
            panic!(
                "The type of the value you provided is not what was expected, {:?}",
                &binding.gtype
            );
        } */

        binding.data = Some(i.data.unwrap());
        binding.length = Some(i.length.unwrap());
    }

    run(program, &in_bindings, out_bindings)
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
