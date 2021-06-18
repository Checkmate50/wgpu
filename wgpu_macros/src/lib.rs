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
