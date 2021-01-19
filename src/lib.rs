#![crate_type = "lib"]
#![feature(const_fn, const_panic)]
#[macro_use]
pub mod debug;
pub mod bind;
pub mod context;
pub mod helper;
pub mod shared;

//pub mod wgpu_compute_header;
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

// hi
/* pub mod bind;

#[macro_use]
extern crate eager;

my_shader! {compute = {
    [[buffer loop in out] uint[]] indices;
    [[buffer in] uint[]] indices2;
    //[[buffer out] uint[]] result;
    //[... uint] xindex;
    {{
        void main() {
            // uint xindex = gl_GlobalInvocationID.x;
            uint index = gl_GlobalInvocationID.x;
            indices[index] = indices[index]+indices2[index];
        }
    }}
}} */
