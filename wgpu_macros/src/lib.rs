extern crate proc_macro;
use proc_macro::TokenStream;

mod get_view;

#[proc_macro]
pub fn create_get_view_func(input: TokenStream) -> TokenStream {
    crate::get_view::sub_module_get_view_func(input)
}

mod generic_bindings;

#[proc_macro]
pub fn generic_bindings(input: TokenStream) -> TokenStream {
    crate::generic_bindings::sub_module_generic_bindings(input)
}

mod graphics_program;

#[proc_macro]
pub fn graphics_program(input: TokenStream) -> TokenStream {
    crate::graphics_program::sub_module_graphics_program(input)
}
