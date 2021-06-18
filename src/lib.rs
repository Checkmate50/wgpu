#![crate_type = "lib"]
#![feature(const_panic)]
#![feature(const_generics)]

use wgpu::BufferSlice;
#[macro_use]
pub mod debug;
pub mod bind;
pub mod helper;
pub mod shared;
pub mod read;
pub mod write;

pub mod wgpu_compute_header;
pub mod wgpu_graphics_header;

// Traits for the proc macros
pub trait AbstractBind {
    fn new() -> Self;
}

pub struct Bound {}

pub struct Unbound {}

impl AbstractBind for Bound {
    fn new() -> Self {
        Bound {}
    }
}

impl AbstractBind for Unbound {
    fn new() -> Self {
        Unbound {}
    }
}

pub trait ContextInputs {
    fn inputs(&self) -> Vec<String>;
}

pub trait RuntimePass<'a> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bindgroup: &'a wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    );

    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>);
}

impl<'a> RuntimePass<'a> for wgpu::RenderPass<'a> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets)
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        self.set_vertex_buffer(slot, buffer_slice)
    }
}
impl<'a> RuntimePass<'a> for wgpu::ComputePass<'a> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets)
    }

    #[allow(unconditional_recursion)]
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        self.set_vertex_buffer(slot, buffer_slice)
    }
}

#[macro_export]
macro_rules! eager_compute_shader {
    ($name:tt!()) => {
        eager! { lazy! {compute_shader! { eager!{$name!()}}}}
    };
}

#[macro_export]
macro_rules! eager_graphics_shader {
    ($name:tt!()) => {
        eager! { lazy! {graphics_shader! { eager!{$name!()}}}}
    };
}

#[macro_export]
macro_rules! eager_binding {
    ($context_name:tt = $($macro_name:tt!()),*) => {eager! { lazy! { wgpu_macros::generic_bindings! { $context_name = eager!{ $($macro_name!()),*}}}}}
}
