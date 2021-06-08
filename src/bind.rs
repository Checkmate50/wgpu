pub use crate::read::MyBufferView;
use crate::shared::{GLSLTYPE, QUALIFIER};
pub use crate::write;
pub use crate::write::MyBufferViewMut;
pub use crate::align::Alignment;
use std::marker::PhantomData;
use wgpu_macros::create_get_view_func;
use zerocopy::AsBytes as _;

use std::rc::Rc;

use wgpu::util::DeviceExt;

/// This trait describes the general methods that are needed to convert a rust type into valid data bound on the device.
pub trait WgpuType {
    /// Sends the data to the device and a handler to that data is returned as `BoundData`.
    #[doc(hidden)]
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData;

    /// This is the size of the type for the purposes of layout
    /// This is not the size of the underlying data
    #[doc(hidden)]
    fn size_of() -> usize;

    /// This is used to convert the compile time type into a valid layout on the device without knowing what the value of the data will be.
    fn create_binding_type() -> wgpu::BindingType;

    /// Sometimes the usage of the underlying data is described by how it is bound. For example, a Vertex will always have `wgpu::BufferUsage::VERTEX`. However, for other buffers the value depends on it's compile time type. For example, whether we are creating a uniform or storage buffer. In this second case, I've added this convenience function to get the appropriate qualifiers from the type.
    fn get_qualifiers() -> Option<QUALIFIER>;
}

/// This struct is used to hold traditional array like data such as single values, vectors, and matricies. It is parameterized on a `BINDINGTYPE` which specifies whether the end buffer is suppose to be a uniform or storage buffer.
pub struct BufferData<const BINDINGTYPE: wgpu::BufferBindingType, T> {
    data: T,
}

impl<const BINDINGTYPE: wgpu::BufferBindingType, T> BufferData<BINDINGTYPE, T> {
    pub fn new(data: T) -> Self {
        BufferData { data }
    }
}

impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType for BufferData<BINDINGTYPE, f32> {
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        BoundData::new_buffer(
            device,
            self.data.align_bytes(),
            1 as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }
    fn size_of() -> usize {
        <f32>::alignment_size()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}

impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType for BufferData<BINDINGTYPE, Vec<u32>> {
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        BoundData::new_buffer(
            device,
            self.data.as_slice().as_bytes(),
            self.data.len() as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }
    fn size_of() -> usize {
        std::mem::size_of::<[u32; 1]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}

impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType for BufferData<BINDINGTYPE, Vec<f32>> {
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        BoundData::new_buffer(
            device,
            self.data.as_slice().as_bytes(),
            self.data.len() as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }
    fn size_of() -> usize {
        std::mem::size_of::<[f32; 1]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}

impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType
    for BufferData<BINDINGTYPE, Vec<[f32; 2]>>
{
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        let numbers: Vec<f32> = self
            .data
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        BoundData::new_buffer(
            device,
            numbers.as_slice().as_bytes(),
            self.data.len() as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 2]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}
impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType
    for BufferData<BINDINGTYPE, Vec<[f32; 3]>>
{
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        let numbers: Vec<f32> = self
            .data
            .clone()
            .into_iter()
            .map(|x| {
                let mut y = x.to_vec();
                y.push(0.0);
                y
            }) // We need to extend Vec3 -> Vec4 for alignment
            .flatten()
            .collect();
        BoundData::new_buffer(
            device,
            numbers.as_slice().as_bytes(),
            self.data.len() as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 4]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}
impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType
    for BufferData<BINDINGTYPE, Vec<[f32; 4]>>
{
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {
        let numbers: Vec<f32> = self
            .data
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        BoundData::new_buffer(
            device,
            numbers.as_slice().as_bytes(),
            self.data.len() as u64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }

    fn size_of() -> usize {
        std::mem::size_of::<[f32; 4]>()
    }
    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}
impl<const BINDINGTYPE: wgpu::BufferBindingType> WgpuType
    for BufferData<BINDINGTYPE, cgmath::Matrix4<f32>>
{
    fn bind(&self, device: &wgpu::Device, qual: Option<QUALIFIER>) -> BoundData {

        BoundData::new_buffer(
            device,
            self.data.align_bytes(),
            64,
            Self::size_of(),
            qual,
            Self::create_binding_type(),
        )
    }

    fn size_of() -> usize {
        <cgmath::Matrix4<f32>>::alignment_size()
    }

    fn create_binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: BINDINGTYPE,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
        }
    }
    fn get_qualifiers() -> Option<QUALIFIER> {
        match BINDINGTYPE {
            wgpu::BufferBindingType::Uniform => Some(QUALIFIER::UNIFORM),
            wgpu::BufferBindingType::Storage { read_only: _ } => Some(QUALIFIER::BUFFER),
        }
    }
}

/// Used to specify https://wgpu.rs/doc/wgpu_types/enum.BindingType.html#variant.Sampler.field.comparison
#[derive(PartialEq, Eq)]
pub enum SamplerComparison {
    True,
    False,
}

/// Used to specify https://wgpu.rs/doc/wgpu_types/enum.BindingType.html#variant.Sampler.field.filtering
#[derive(PartialEq, Eq)]
pub enum SamplerFiltering {
    True,
    False,
}

/// This struct is used to create samplers for a pipeline. It is parameterized on two enums which specify whether the sampler created should be a comparison sampler and whether it should be a filtering sampler.
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
    fn bind(&self, device: &wgpu::Device, _: Option<QUALIFIER>) -> BoundData {
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
    fn get_qualifiers() -> Option<QUALIFIER> {
        //todo Do I need qualifiers here?
        None
    }
}

/// Used to specify https://docs.rs/wgpu/0.8.1/wgpu/enum.BindingType.html#variant.Texture.field.multisampled
#[derive(PartialEq, Eq)]
pub enum TextureMultisampled {
    True,
    False,
}

/// This struct is
/// For `SAMPLETYPE` https://docs.rs/wgpu/0.8.1/wgpu/enum.BindingType.html#variant.Texture.field.sample_type
/// For `VIEWDIMENSION` https://docs.rs/wgpu/0.8.1/wgpu/enum.BindingType.html#variant.Texture.field.view_dimension
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
    fn bind(&self, device: &wgpu::Device, _: Option<QUALIFIER>) -> BoundData {
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
    fn get_qualifiers() -> Option<QUALIFIER> {
        //todo Do I need qualifiers here?
        None
    }
}

#[doc(hidden)]
/// The result of binding WGPUType data to the gpu. These are basically all handlers to GPU data of different types.
pub enum BoundData {
    Buffer {
        data: Rc<wgpu::Buffer>,
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
    pub fn new_buffer(
        device: &wgpu::Device,
        data: &[u8],
        length: u64,
        size: usize,
        qual: Option<QUALIFIER>,
        binding_type: wgpu::BindingType,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: match qual {
                Some(QUALIFIER::VERTEX) => wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
                Some(QUALIFIER::UNIFORM) => {
                    wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST
                }
                Some(QUALIFIER::BUFFER) | Some(_) | None => {
                    wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC
                }
            },
        });

        BoundData::Buffer {
            data: Rc::new(buffer),
            len: length,
            size,
            binding_type,
        }
    }

    pub fn get_buffer_size_bytes(&self) -> Option<u64> {
        match self {
            BoundData::Buffer { len, size, .. } => Some(*len * *size as u64),
            _ => None,
        }
    }

    pub fn get_buffer(&self) -> Option<(Rc<wgpu::Buffer>, u64, usize)> {
        match self {
            BoundData::Buffer {
                data, len, size, ..
            } => Some((data.clone(), *len, *size)),
            _ => None,
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

//todo get rid of this when there is a new way to stringify shaders
#[derive(Debug, Clone)]
pub struct TextureBinding {
    pub binding_number: u32,
    pub group_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::TextureView>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

//todo get rid of this when there is a new way to stringify shaders
#[derive(Debug, Clone)]
pub struct SamplerBinding {
    pub binding_number: u32,
    pub group_number: u32,
    pub name: String,
    pub data: Option<Rc<wgpu::Sampler>>,
    pub gtype: GLSLTYPE,
    pub qual: Vec<QUALIFIER>,
}

//todo get rid of this when there is a new way to stringify shaders
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

/// A buffer of indices that can be used to run a pipeline by indexing into the vertex buffer(s) instead of iterating over them
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

/// A Vertex Buffer containing data of type WgpuType
pub struct Vertex<A: WgpuType + ?Sized> {
    typ: PhantomData<A>,
    buffer: Rc<wgpu::Buffer>,
}

impl<'a, A: WgpuType> Vertex<A> {
    pub fn get_buffer(&'a self) -> &'a wgpu::Buffer {
        &self.buffer
    }

    pub fn size_of() -> usize {
        A::size_of()
    }

    pub fn new(device: &wgpu::Device, data: &A) -> Self {
        Vertex {
            typ: PhantomData,
            buffer: data
                .bind(device, Some(QUALIFIER::VERTEX))
                .get_buffer()
                .unwrap()
                .0,
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
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // can check which stage
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

//todo also do impl's with macro
impl<'a, B: WgpuType> BindGroup1<B> {
    pub fn get_bind_group(&'a self) -> &'a wgpu::BindGroup {
        &self.bind_group
    }

    pub fn get_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let bind_entry_vec = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX
                | wgpu::ShaderStage::FRAGMENT
                | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
            ty: B::create_binding_type(),
            count: None,
        }];

        //debug!(bind_entry_vec);

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
            entries: &bind_entry_vec,
            label: None,
        })
    }

    pub fn new(device: &wgpu::Device, data0: &B) -> Self {
        let data = vec![data0.bind(device, B::get_qualifiers())];

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
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: B::create_binding_type(),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: C::create_binding_type(),
                count: None,
            },
        ];

        //debug!(bind_entry_vec);

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            // The layout of for each binding specify a number to connect with the bind_group, a visibility to specify for which stage it's for and a type
            entries: &bind_entry_vec,
            label: None,
        })
    }

    pub fn new(device: &wgpu::Device, data0: &B, data1: &C) -> Self {
        let data = vec![
            data0.bind(device, B::get_qualifiers()),
            data1.bind(device, B::get_qualifiers()),
        ];

        let bind_group = create_bind_group(device, &data);

        BindGroup2 {
            typ1: PhantomData,
            typ2: PhantomData,
            data,
            bind_group,
        }
    }
    pub fn get_buffers(&self) -> Vec<Option<Rc<wgpu::Buffer>>> {
        self.data
            .iter()
            .map(|d| d.get_buffer().and_then(|x| Some(x.0)))
            .collect()
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
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: B::create_binding_type(),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
                ty: C::create_binding_type(),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::VERTEX
                    | wgpu::ShaderStage::FRAGMENT
                    | wgpu::ShaderStage::COMPUTE, // TODO I've made uniforms visible to both stages even if they are only used in one. Find out if this is a bad thing
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
            data0.bind(device, B::get_qualifiers()),
            data1.bind(device, B::get_qualifiers()),
            data2.bind(device, B::get_qualifiers()),
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
    pub fn get_buffers(&self) -> Vec<Option<Rc<wgpu::Buffer>>> {
        self.data
            .iter()
            .map(|d| d.get_buffer().and_then(|x| Some(x.0)))
            .collect()
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

//todo redo as a macro like get_view
impl<'a, const BINDINGTYPE1: wgpu::BufferBindingType, T> BindGroup1<BufferData<BINDINGTYPE1, T>>
where
    BufferData<BINDINGTYPE1, T>: WgpuType,
{
    pub fn setup_read_0(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferView {
        MyBufferView::new(
            device,
            encoder,
            &self.data.get(0).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }

    pub fn setup_write_0(
        &self,
        device: &wgpu::Device,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferViewMut {
        MyBufferViewMut::new(
            device,
            self.data.get(0).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }
}

impl<'a, const BINDINGTYPE: wgpu::BufferBindingType, T, R> BindGroup2<BufferData<BINDINGTYPE, T>, R>
where
    BufferData<BINDINGTYPE, T>: WgpuType,
    R: WgpuType,
{
    pub fn setup_read_0(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferView {
        MyBufferView::new(
            device,
            encoder,
            &self.data.get(0).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }

    pub fn setup_write_0(
        &self,
        device: &wgpu::Device,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferViewMut {
        MyBufferViewMut::new(
            device,
            self.data.get(0).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }
}

impl<'a, const BINDINGTYPE: wgpu::BufferBindingType, T, R> BindGroup2<R, BufferData<BINDINGTYPE, T>>
where
    BufferData<BINDINGTYPE, T>: WgpuType,
    R: WgpuType,
{
    pub fn setup_read_1(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferView {
        MyBufferView::new(
            device,
            encoder,
            &self.data.get(1).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }

    pub fn setup_write_1(
        &self,
        device: &wgpu::Device,
        bounds: std::ops::Range<u64>,
    ) -> MyBufferViewMut {
        MyBufferViewMut::new(
            device,
            self.data.get(1).unwrap().get_buffer().unwrap().0,
            bounds,
        )
    }
}
