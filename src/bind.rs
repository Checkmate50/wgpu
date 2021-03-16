use crate::shared::{GLSLTYPE, QUALIFIER};
use core::panic;
use std::marker::PhantomData;
use wgpu_macros::create_get_view_func;
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
        bind_helper(
            device,
            self.as_bytes(),
            1,
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
        )
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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
            Self::size_of(),
            Some(qual),
            Self::create_binding_type(),
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

#[derive(PartialEq, Eq)]
pub enum SamplerComparison {
    True,
    False,
}

#[derive(PartialEq, Eq)]
pub enum SamplerFiltering {
    True,
    False,
}

pub struct SamplerData<'a, const COMPARABLE: SamplerComparison, const FILTERABLE: SamplerFiltering>
{
    desc: wgpu::SamplerDescriptor<'a>,
}

impl<'a, const COMPARABLE: SamplerComparison, const FILTERABLE: SamplerFiltering>
    SamplerData<'a, COMPARABLE, FILTERABLE>
{
    pub fn new(desc: wgpu::SamplerDescriptor<'a>) -> Self {
        if (desc.compare.is_some() && COMPARABLE == SamplerComparison::False)
            || (desc.compare.is_none() && COMPARABLE == SamplerComparison::True)
        {
            panic!("The descriptor compare field does not match up with the type of this SamplerData struct")
        };
        SamplerData { desc }
    }
}

impl<'a, const COMPARABLE: SamplerComparison, const FILTERABLE: SamplerFiltering> WgpuType
    for SamplerData<'a, COMPARABLE, FILTERABLE>
{
    fn bind(&self, device: &wgpu::Device, _: QUALIFIER) -> BoundData {
        BoundData::Sampler {
            data: device.create_sampler(&self.desc),
            binding_type: Self::create_binding_type(),
        }
    }
    fn size_of() -> usize {
        panic!("TODO: I haven't checked the size of this yet")
    }

    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Sampler {
            comparison: COMPARABLE == SamplerComparison::True,
            filtering: FILTERABLE == SamplerFiltering::True,
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum TextureMultisampled {
    True,
    False,
}

pub struct TextureData<
    'a,
    const MULTISAMPLE: TextureMultisampled,
    const SAMPLETYPE: wgpu::TextureSampleType,
    const VIEWDIMENSION: wgpu::TextureViewDimension,
> {
    data: Option<Vec<u8>>,
    desc: wgpu::TextureDescriptor<'a>,
    view_desc: wgpu::TextureViewDescriptor<'a>,
    queue: Rc<wgpu::Queue>,
}

impl<
        'a,
        const MULTISAMPLE: TextureMultisampled,
        const SAMPLETYPE: wgpu::TextureSampleType,
        const VIEWDIMENSION: wgpu::TextureViewDimension,
    > TextureData<'a, MULTISAMPLE, SAMPLETYPE, VIEWDIMENSION>
{
    pub fn new(
        data: Vec<u8>,
        desc: wgpu::TextureDescriptor<'a>,
        view_desc: wgpu::TextureViewDescriptor<'a>,
        queue: Rc<wgpu::Queue>,
    ) -> Self {
        //todo check the view dimension lines up
        TextureData {
            data: Some(data),
            desc,
            view_desc,
            queue,
        }
    }
    pub fn new_without_data(
        desc: wgpu::TextureDescriptor<'a>,
        view_desc: wgpu::TextureViewDescriptor<'a>,
        queue: Rc<wgpu::Queue>,
    ) -> Self {
        //todo check the view dimension lines up
        TextureData {
            data: None,
            desc,
            view_desc,
            queue,
        }
    }
}

impl<
        'a,
        const MULTISAMPLE: TextureMultisampled,
        const SAMPLETYPE: wgpu::TextureSampleType,
        const VIEWDIMENSION: wgpu::TextureViewDimension,
    > WgpuType for TextureData<'a, MULTISAMPLE, SAMPLETYPE, VIEWDIMENSION>
{
    fn bind(&self, device: &wgpu::Device, _: QUALIFIER) -> BoundData {
        let texture = match &self.data {
            Some(data) => device.create_texture_with_data(&self.queue, &self.desc, data),
            None => device.create_texture(&self.desc),
        };
        let view = texture.create_view(&self.view_desc);
        BoundData::Texture {
            data: texture,
            view,
            binding_type: Self::create_binding_type(),
        }
    }
    fn size_of() -> usize {
        panic!("TODO: I haven't checked the size of this yet")
    }

    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            multisampled: MULTISAMPLE == TextureMultisampled::True,
            sample_type: SAMPLETYPE,
            view_dimension: VIEWDIMENSION,
        }
    }
}

pub enum BoundData {
    Buffer {
        data: wgpu::Buffer,
        len: u64,
        size: usize,
        binding_type: wgpu::BindingType,
    },
    Texture {
        data: wgpu::Texture,
        view: wgpu::TextureView,
        binding_type: wgpu::BindingType,
    },
    Sampler {
        data: wgpu::Sampler,
        binding_type: wgpu::BindingType,
    },
}

impl BoundData {
    pub fn get_buffer(self) -> (wgpu::Buffer, u64, usize) {
        match self {
            BoundData::Buffer {
                data, len, size, ..
            } => (data, len, size),
            _ => unreachable!(),
        }
    }
    pub fn get_texture(self) -> wgpu::TextureView {
        match self {
            BoundData::Texture { view, .. } => view,
            _ => unreachable!(),
        }
    }
    pub fn get_sampler(self) -> wgpu::Sampler {
        match self {
            BoundData::Sampler { data, .. } => (data),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextureBinding {
    pub binding_number: u32,
    pub group_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::TextureView>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

#[derive(Debug, Clone)]
pub struct SamplerBinding {
    pub binding_number: u32,
    pub group_number: u32,
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
    size: usize,
    qual: Option<QUALIFIER>,
    binding_type: wgpu::BindingType,
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
        size,
        binding_type,
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

fn create_bind_group(device: &wgpu::Device, buffers: &Vec<BoundData>) -> wgpu::BindGroup {
    let bind_entry: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| {
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // can check which stage
                ty: match buf {
                    BoundData::Buffer { binding_type, .. } => *binding_type,
                    //todo check that these are good
                    BoundData::Texture { binding_type, .. } => *binding_type,
                    BoundData::Sampler { binding_type, .. } => *binding_type,
                },

                count: None,
            }
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
            resource: match buf {
                BoundData::Buffer { data, .. } => data.as_entire_binding(),
                BoundData::Texture { view, .. } => wgpu::BindingResource::TextureView(&view),
                BoundData::Sampler { data, .. } => wgpu::BindingResource::Sampler(data),
            },
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
    data: Vec<BoundData>,
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
        let data = vec![data0.bind(device, QUALIFIER::UNIFORM)];

        let bind_group = create_bind_group(device, &data);

        BindGroup1 {
            typ1: PhantomData,
            data,
            bind_group,
        }
    }
}

pub struct BindGroup2<B: WgpuType, C: WgpuType> {
    typ1: PhantomData<B>,
    typ2: PhantomData<C>,
    data: Vec<BoundData>,
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
        let data = vec![
            data0.bind(device, QUALIFIER::UNIFORM),
            data1.bind(device, QUALIFIER::UNIFORM),
        ];

        let bind_group = create_bind_group(device, &data);

        BindGroup2 {
            typ1: PhantomData,
            typ2: PhantomData,
            data,
            bind_group,
        }
    }
}

pub struct BindGroup3<B: WgpuType, C: WgpuType, D: WgpuType> {
    typ1: PhantomData<B>,
    typ2: PhantomData<C>,
    typ3: PhantomData<D>,
    data: Vec<BoundData>,
    bind_group: wgpu::BindGroup,
}

impl<'a, B: WgpuType, C: WgpuType, D: WgpuType> BindGroup3<B, C, D> {
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: D::create_binding_type(),
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

    /// Initializes data on the device and returns it as a group
    pub fn new(device: &wgpu::Device, data0: &B, data1: &C, data2: &D) -> Self {
        let data = vec![
            data0.bind(device, QUALIFIER::UNIFORM),
            data1.bind(device, QUALIFIER::UNIFORM),
            data2.bind(device, QUALIFIER::UNIFORM),
        ];

        let bind_group = create_bind_group(device, &data);

        BindGroup3 {
            typ1: PhantomData,
            typ2: PhantomData,
            typ3: PhantomData,
            data,
            bind_group,
        }
    }
}

/* impl<
        'a,
        const MULTISAMPLE: TextureMultisampled,
        const SAMPLETYPE: wgpu::TextureSampleType,
        const VIEWDIMENSION: wgpu::TextureViewDimension,
        C: WgpuType,
        D: WgpuType,
    > BindGroup3<TextureData<'a, MULTISAMPLE, SAMPLETYPE, VIEWDIMENSION>, C, D>
{
    pub fn get_view_0(self, desc: &wgpu::TextureViewDescriptor) -> wgpu::TextureView {
        match self.data.get(0).unwrap() {
            BoundData::Texture { data, .. } => data.create_view(desc),
            _ => panic!("not a texture"),
        }
    }
} */

create_get_view_func!(1);
create_get_view_func!(2);
create_get_view_func!(3);
