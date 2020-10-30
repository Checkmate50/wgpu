use zerocopy::AsBytes as _;

use crate::shared::{GLSLTYPE, QUALIFIER};

use std::rc::Rc;

#[macro_export]
macro_rules! bind {
    ($device:tt, $bindings:tt, $out_bindings:tt, $name:tt, $data:tt, $context:tt, $bind_context:tt) => {{
        if $bind_context.do_consume {
            panic!("You need to use bind_consume here")
        }
        Bindable::bind(
            &$data,
            &$device,
            &mut $bindings,
            &mut $out_bindings,
            $name.to_string(),
            &$context,
        )
    }};
}

#[macro_export]
macro_rules! bind_consume {
    ($device:tt, $bindings:tt, $out_bindings:tt, $name:tt, $data:tt, $context:tt, $bind_context:tt) => {{
        if !$bind_context.do_consume {
            panic!("You should be using bind here")
        }
        Bindable::bind_consume(
            &$data,
            &$device,
            &mut $bindings,
            &mut $out_bindings,
            $name.to_string(),
            $context,
        )
    }};
}

/* pub fn bind_vec2(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    vecs: &Vec<[f32; 2]>,
    name: String,
) {
    let numbers: Vec<f32> = vecs
        .clone()
        .into_iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect();
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        vecs.len() as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
} */

/* pub fn bind_fvec2(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &Vec<f32>,
    name: String,
) {
    if numbers.len() % 2 != 0 {
        panic!(
            "Your trying to bind to vec to but your not giving a vector that can be split into 2's"
        )
    }
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        (numbers.len() / 2) as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
} */

/* pub fn bind_float(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &f32,
    name: String,
) {
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_bytes(),
        1 as u64,
        vec![GLSLTYPE::Float],
        name,
    )
} */

#[derive(Debug)]
pub struct TextureBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::TextureView>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

#[derive(Debug)]
pub struct SamplerBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::Sampler>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

#[derive(Debug)]
pub struct DefaultBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::Buffer>>,
    pub length: Option<u64>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

pub fn new_bindings(bindings: &Vec<DefaultBinding>) -> Vec<DefaultBinding> {
    let mut new = Vec::new();

    for i in bindings.iter() {
        new.push(DefaultBinding {
            name: i.name.to_string(),
            binding_number: i.binding_number,
            qual: i.qual.clone(),
            gtype: i.gtype.clone(),
            data: None,
            length: None,
        })
    }
    new
}

pub trait ProgramBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding>;
    fn get_samplers(&mut self) -> Option<&mut Vec<SamplerBinding>>;
    fn get_textures(&mut self) -> Option<&mut Vec<TextureBinding>>;
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding;
}

pub trait OutProgramBindings {
    fn get_bindings(&mut self) -> &mut Vec<DefaultBinding>;
    fn index_binding(&mut self, index: usize) -> &mut DefaultBinding;
}

pub trait Bindings {
    fn clone(&self) -> Self;
}

fn bind_helper<R: ProgramBindings, T: OutProgramBindings>(
    device: &wgpu::Device,
    bindings: &mut R,
    out_bindings: &mut T,
    data: &[u8],
    length: u64,
    acceptable_types: Vec<GLSLTYPE>,
    name: String,
) {
    let mut binding = match bindings.get_bindings().iter().position(|x| x.name == name) {
        Some(x) => bindings.index_binding(x),
        None => {
            let x = out_bindings
                .get_bindings()
                .iter()
                .position(|x| x.name == name)
                .expect(&format!("We couldn't find the binding for {}", name));
            out_bindings.index_binding(x)
        }
    };

    if !acceptable_types.contains(&binding.gtype) {
        println!("{:?}", &binding.name);
        println!("{:?}", acceptable_types);
        panic!(
            "The type of the value you provided is not what was expected, {:?}",
            &binding.gtype
        );
    }

    let buffer = device.create_buffer_with_data(
        data,
        if binding.qual.contains(&QUALIFIER::VERTEX) {
            wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST
        } else if binding.qual.contains(&QUALIFIER::UNIFORM) {
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST
        } else {
            wgpu::BufferUsage::MAP_READ
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_SRC
                | wgpu::BufferUsage::VERTEX
        },
    );

    binding.data = Some(Rc::new(buffer));
    binding.length = Some(length);
}

pub trait Bindable {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    );
}

impl Bindable for Vec<u32> {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        bind_helper(
            device,
            bindings,
            out_bindings,
            self.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::ArrayInt, GLSLTYPE::ArrayUint],
            name,
        );
    }
}

impl Bindable for Vec<f32> {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        bind_helper(
            device,
            bindings,
            out_bindings,
            self.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::Float, GLSLTYPE::ArrayFloat],
            name,
        )
    }
}

impl Bindable for Vec<[f32; 3]> {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        bind_helper(
            device,
            bindings,
            out_bindings,
            numbers.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::Vec3, GLSLTYPE::ArrayVec3],
            name,
        )
    }
}
impl Bindable for Vec<[f32; 4]> {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        bind_helper(
            device,
            bindings,
            out_bindings,
            numbers.as_slice().as_bytes(),
            self.len() as u64,
            vec![GLSLTYPE::Vec4, GLSLTYPE::ArrayVec4],
            name,
        );
    }
}

impl Bindable for wgpu::SamplerDescriptor {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        let mut binding = match bindings
            .get_samplers()
            .unwrap()
            .iter()
            .position(|x| x.name == name)
        {
            Some(x) => &mut bindings.get_samplers().unwrap()[x],
            None => {
                panic!("I haven't considered that you would output a sampler yet")
                /* let x = out_bindings
                    .getBindings()
                    .iter()
                    .position(|x| x.name == name)
                    .expect("We couldn't find the binding");
                out_bindings.samplers[x] */
            }
        };
        binding.data = Some(Rc::new(device.create_sampler(self)));
    }
}

impl Bindable for wgpu::Texture {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        let mut binding = match bindings
            .get_textures()
            .unwrap()
            .iter()
            .position(|x| x.name == name)
        {
            Some(x) => &mut bindings.get_textures().unwrap()[x],
            None => {
                panic!("I haven't considered that you would output a texture yet")
                /* let x = out_bindings
                    .getBindings()
                    .iter()
                    .position(|x| x.name == name)
                    .expect("We couldn't find the binding");
                out_bindings.samplers[x] */
            }
        };
        binding.data = Some(Rc::new(self.create_default_view()));
    }
}

impl Bindable for cgmath::Matrix4<f32> {
    fn bind<R: ProgramBindings, T: OutProgramBindings>(
        &self,
        device: &wgpu::Device,
        bindings: &mut R,
        out_bindings: &mut T,
        name: String,
    ) {
        let mat_slice: &[f32; 16] = self.as_ref();
        bind_helper(
            device,
            bindings,
            out_bindings,
            bytemuck::cast_slice(mat_slice.as_bytes()),
            64 as u64,
            vec![GLSLTYPE::Mat4],
            name,
        );
    }
}

/* pub fn bind_vec2(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    vecs: &Vec<[f32; 2]>,
    name: String,
) {
    let numbers: Vec<f32> = vecs
        .clone()
        .into_iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect();
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        vecs.len() as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
} */

/* pub fn bind_fvec2(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &Vec<f32>,
    name: String,
) {
    if numbers.len() % 2 != 0 {
        panic!(
            "Your trying to bind to vec to but your not giving a vector that can be split into 2's"
        )
    }
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_slice().as_bytes(),
        (numbers.len() / 2) as u64,
        if numbers.len() == 2 {
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        } else {
            //todo only ArrayVec
            vec![GLSLTYPE::Vec2, GLSLTYPE::ArrayVec2]
        },
        name,
    )
} */

/* pub fn bind_mat4(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    mat: cgmath::Matrix4<f32>,
    name: String,
) {
    let mat_slice: &[f32; 16] = mat.as_ref();
    bind_helper(
        program,
        bindings,
        out_bindings,
        bytemuck::cast_slice(mat_slice.as_bytes()),
        64 as u64,
        vec![GLSLTYPE::Mat4],
        name,
    )
} */

/* pub fn bind_float(
    device: wgpu::Device,
    bindings: &mut dyn ProgramBindings,
    out_bindings: &mut dyn OutProgramBindings,
    numbers: &f32,
    name: String,
) {
    bind_helper(
        program,
        bindings,
        out_bindings,
        numbers.as_bytes(),
        1 as u64,
        vec![GLSLTYPE::Float],
        name,
    )
} */
