use crate::shared::{GLSLTYPE, QUALIFIER};
use std::marker::PhantomData;
use zerocopy::AsBytes as _;

use std::rc::Rc;

use wgpu::util::DeviceExt;

pub trait WgpuType {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData;

    // This is the size of the type for the purposes of layout
    // This is not the size of the underlying data
    fn size_of() -> usize;

    fn create_binding_type() -> wgpu::BindingType;
}

impl WgpuType for f32 {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        bind_helper(device, self.as_bytes(), 1, Some(qual))
    }
    fn size_of() -> usize {
        std::mem::size_of::<f32>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for Vec<u32> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        bind_helper(
            device,
            self.as_slice().as_bytes(),
            self.len() as u64,
            Some(qual),
        )
    }
    fn size_of() -> usize {
        std::mem::size_of::<[u32; 1]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for Vec<f32> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        bind_helper(
            device,
            self.as_slice().as_bytes(),
            self.len() as u64,
            Some(qual),
        )
    }
    fn size_of() -> usize {
        std::mem::size_of::<[f32; 1]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for Vec<[f32; 2]> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        bind_helper(
            device,
            numbers.as_slice().as_bytes(),
            self.len() as u64,
            Some(qual),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 2]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for Vec<[f32; 3]> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        bind_helper(
            device,
            numbers.as_slice().as_bytes(),
            self.len() as u64,
            Some(qual),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 3]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for Vec<[f32; 4]> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        let numbers: Vec<f32> = self
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        bind_helper(
            device,
            numbers.as_slice().as_bytes(),
            self.len() as u64,
            Some(qual),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 4]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for cgmath::Matrix4<f32> {
    fn bind(&self, device: &wgpu::Device, qual: QUALIFIER) -> BoundData {
        let mat_slice: &[f32; 16] = self.as_ref();
        bind_helper(
            device,
            bytemuck::cast_slice(mat_slice.as_bytes()),
            64,
            Some(qual),
        )
    }

    fn size_of() -> usize {
        64
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
}

impl WgpuType for wgpu::SamplerDescriptor<'_> {
    fn bind(&self, device: &wgpu::Device, _: QUALIFIER) -> BoundData {
        BoundData::Sampler {
            data: device.create_sampler(self),
        }
    }
    fn size_of() -> usize {
        panic!("TODO: I haven't checked the size of this yet")
    }

    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Sampler {
            comparison: false,
            filtering: true,
        }
    }
}

impl WgpuType for wgpu::Texture {
    fn bind(&self, _device: &wgpu::Device, _: QUALIFIER) -> BoundData {
        BoundData::Texture {
            data: self.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
    fn size_of() -> usize {
        panic!("TODO: I haven't checked the size of this yet")
    }

    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        }
    }
}

pub enum BoundData {
    Buffer { data: wgpu::Buffer, len: u64 },
    Texture { data: wgpu::TextureView },
    Sampler { data: wgpu::Sampler },
}

impl BoundData {
    pub fn get_buffer(self) -> (wgpu::Buffer, u64) {
        match self {
            BoundData::Buffer { data, len } => (data, len),
            _ => unreachable!(),
        }
    }
    pub fn get_texture(self) -> wgpu::TextureView {
        match self {
            BoundData::Texture { data } => data,
            _ => unreachable!(),
        }
    }
    pub fn get_sampler(self) -> wgpu::Sampler {
        match self {
            BoundData::Sampler { data } => (data),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextureBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::TextureView>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

#[derive(Debug, Clone)]
pub struct SamplerBinding {
    pub binding_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::Sampler>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

#[derive(Debug, Clone)]
pub struct DefaultBinding {
    pub binding_number: u32,
    pub group_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::Buffer>>,
    pub length: Option<u64>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

pub struct Indices {
    pub buffer: Rc<wgpu::Buffer>,
    pub len: u32,
}

impl Indices {
    pub fn new(device: &wgpu::Device, data: &Vec<u16>) -> Self {
        Indices {
            buffer: Rc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data.as_slice().as_bytes(),
                    usage: wgpu::BufferUsage::INDEX,
                }),
            ),
            len: data.len() as u32,
        }
    }
}

fn bind_helper(
    device: &wgpu::Device,
    data: &[u8],
    length: u64,
    qual: Option<QUALIFIER>,
) -> BoundData {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: match qual {
            Some(QUALIFIER::VERTEX) => wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            Some(QUALIFIER::UNIFORM) => wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            Some(_) | None => {
                wgpu::BufferUsage::MAP_READ
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_SRC
                    | wgpu::BufferUsage::VERTEX
            }
        },
    });

    BoundData::Buffer {
        data: buffer,
        len: length,
    }
}

pub struct Vertex<A: WgpuType + ?Sized> {
    typ: PhantomData<A>,
    buffer: wgpu::Buffer,
}

impl<'a, A: WgpuType> Vertex<A> {
    pub fn get_buffer(&'a self) -> &'a wgpu::Buffer {
        &self.buffer
    }

    pub fn new(device: &wgpu::Device, data: &A) -> Self {
        Vertex {
            typ: PhantomData,
            buffer: data.bind(device, QUALIFIER::VERTEX).get_buffer().0,
        }
    }
}

fn create_bind_group(device: &wgpu::Device, buffers: Vec<(wgpu::Buffer, u64)>) -> wgpu::BindGroup {
    let bind_entry: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // can check which stage
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(buf.1),
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &bind_entry,
        label: None,
    });

    let bind_group_bindings: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.0.as_entire_binding(),
        })
        .collect();
    let bgd = &wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: bind_group_bindings.as_slice(),
        label: None,
    };
    device.create_bind_group(bgd)
}

// todo generate with a macro
pub struct BindGroup1<B: WgpuType> {
    typ1: PhantomData<B>,
    bind_group: wgpu::BindGroup,
}

impl<'a, B: WgpuType> BindGroup1<B> {
    pub fn get_bind_group(&'a self) -> &'a wgpu::BindGroup {
        &self.bind_group
    }

    pub fn get_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let bind_entry_vec = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
            ty: B::create_binding_type(),
            count: None,
        }];

        debug!(bind_entry_vec);

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
            entries: &bind_entry_vec,
            label: None,
        })
    }

    pub fn new(device: &wgpu::Device, data0: &B) -> Self {
        let buffers = vec![data0.bind(device, QUALIFIER::UNIFORM).get_buffer()];

        BindGroup1 {
            typ1: PhantomData,
            bind_group: create_bind_group(device, buffers),
        }
    }
}

pub struct BindGroup2<B: WgpuType, C: WgpuType> {
    typ1: PhantomData<B>,
    typ2: PhantomData<C>,

    bind_group: wgpu::BindGroup,
}

impl<'a, B: WgpuType, C: WgpuType> BindGroup2<B, C> {
    pub fn get_bind_group(&'a self) -> &'a wgpu::BindGroup {
        &self.bind_group
    }

    pub fn get_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let bind_entry_vec = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: B::create_binding_type(),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: C::create_binding_type(),
                count: None,
            },
        ];

        debug!(bind_entry_vec);

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
            entries: &bind_entry_vec,
            label: None,
        })
    }

    pub fn new(device: &wgpu::Device, data0: &B, data1: &C) -> Self {
        let buffers = vec![
            data0.bind(device, QUALIFIER::UNIFORM).get_buffer(),
            data1.bind(device, QUALIFIER::UNIFORM).get_buffer(),
        ];

        BindGroup2 {
            typ1: PhantomData,
            typ2: PhantomData,
            bind_group: create_bind_group(device, buffers),
        }
    }
}

pub struct BindGroup3<B: WgpuType, C: WgpuType, D: WgpuType> {
    typ1: PhantomData<B>,
    typ2: PhantomData<C>,
    typ3: PhantomData<D>,

    bind_group: wgpu::BindGroup,
}

impl<'a, B: WgpuType, C: WgpuType, D: WgpuType> BindGroup3<B, C, D> {
    pub fn get_bind_group(&'a self) -> &'a wgpu::BindGroup {
        &self.bind_group
    }
    pub fn new(device: &wgpu::Device, data0: &B, data1: &C, data2: &D) -> Self {
        let buffers = vec![
            data0.bind(device, QUALIFIER::UNIFORM).get_buffer(),
            data1.bind(device, QUALIFIER::UNIFORM).get_buffer(),
            data2.bind(device, QUALIFIER::UNIFORM).get_buffer(),
        ];

        BindGroup3 {
            typ1: PhantomData,
            typ2: PhantomData,
            typ3: PhantomData,
            bind_group: create_bind_group(device, buffers),
        }
    }
}
