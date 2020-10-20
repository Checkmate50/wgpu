#![crate_type = "lib"]
#![feature(const_fn, const_panic)]
#![feature(const_mut_refs, unsized_locals)]
#[macro_use]
pub mod debug;
pub mod context;
pub mod helper;
pub mod shared;

pub mod wgpu_compute_header;
pub mod wgpu_graphics_header;

#[macro_export]
macro_rules! eager_compute_shader {
    ($name:tt!()) => {eager! { lazy! {compute_shader! { eager!{$name!()}}}}}
}

#[macro_export]
macro_rules! eager_graphics_shader {
    ($name:tt!()) => {eager! { lazy! {graphics_shader! { eager!{$name!()}}}}}
}

#[macro_export]
macro_rules! eager_binding {
    ($context_name:tt = $($macro_name:tt!()),*) => {eager! { lazy! { generic_bindings! { $context_name = eager!{ $($macro_name!()),*}}}}}
}

// hi
pub mod bind;

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
}}
