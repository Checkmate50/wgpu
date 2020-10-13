extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Ident, Token};

use std::collections::HashSet;
use std::iter;

use rand::Rng;

struct Parameters {
    context: Ident,
    params: HashSet<Ident>,
    outs: HashSet<Ident>,
}

impl Parse for Parameters {
    fn parse(input: ParseStream) -> Result<Self> {
        let context = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let params = Punctuated::<Ident, Token![,]>::parse_separated_nonempty(input)?;
        input.parse::<Token![;]>()?;
        let outs = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        Ok(Parameters {
            context,
            params: params.into_iter().collect(),
            outs: outs.into_iter().collect(),
        })
    }
}

#[proc_macro]
pub fn init(_: TokenStream) -> TokenStream {
    TokenStream::from(quote! {
        use pipeline::shared::{Program};
        use pipeline::bind::{ProgramBindings, OutProgramBindings, Bindable};

        trait AbstractBind {
            fn new() -> Self;
        }

        struct Bound {}

        struct Unbound {}

        impl AbstractBind for Bound {
            fn new() -> Self {
                Bound {}
            }
        }

        impl AbstractBind for Unbound{
            fn new() -> Self {
                Unbound {}
            }
        }

        trait ContextInputs {
            fn inputs(&self) -> Vec<String>;
        }
    })
}

// Implementation 3
#[proc_macro]
pub fn generic_bindings(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as Parameters);

    let input_vec: Vec<Ident> = input.params.into_iter().collect();
    let out_vec: Vec<Ident> = input.outs.into_iter().collect();

    let mut rng = rand::thread_rng();

    let n1: u8 = rng.gen();
    let context = format_ident!("Context{}", n1);

    let mut all_expanded = Vec::new();

    let variables: Vec<Ident> = (1..input_vec.len() + 1)
        .into_iter()
        .map(|x| format_ident!("T{}", x))
        .collect();
    let fields: Vec<Ident> = (1..input_vec.len() + 1)
        .into_iter()
        .map(|x| format_ident!("field{}", x))
        .collect();
    let init: Vec<Ident> = iter::repeat(format_ident!("Unbound"))
        .take(input_vec.len())
        .collect();
    let run: Vec<Ident> = iter::repeat(format_ident!("Bound"))
        .take(input_vec.len())
        .collect();
    let ctxloc = input.context;
    all_expanded.push(quote! {
        struct #context<#(#variables: AbstractBind),*> {
            #(#fields: #variables,)*
        }

        impl #context<#(#init),*> {
            fn new() -> Self {
                #context {
                    #(#fields: Unbound {},)*
                }
            }
        }

        impl ContextInputs for #context<#(#init),*> {
            fn inputs(&self) -> Vec<String> {
                vec![#(stringify!(#input_vec).to_string()),*]
            }
        }

        impl #context<#(#run),*> {
            fn runable<P, B>(&self, f: P) -> B where P: FnOnce() -> B{
                f()
            }
            fn can_pipe(&self, b : &dyn ContextInputs) {
                let a = vec![#(stringify!(#out_vec).to_string()),*];
                assert!(b.inputs().iter().all(|item| a.contains(item)));
            }
        }

        let #ctxloc = #context::new();
    });

    let bound = format_ident!("Bound");
    let unbound = format_ident!("Unbound");

    for i in 0..input_vec.len() {
        let trait_name = format_ident!("BindField{}{}", i + 1, n1);
        let name = format_ident!("{}", input_vec[i]);
        let bind_name = format_ident!("bind_{}", input_vec[i]);
        let mut type_params = variables.clone();
        type_params.remove(i);
        type_params.insert(i, bound.clone());

        let mut trait_params = variables.clone();
        trait_params.remove(i);
        let mut impl_params = variables.clone();
        impl_params.remove(i);
        impl_params.insert(i, unbound.clone());

        // A copy of the input vec with the current param being bound removed so that the names match up with trait_params.
        let mut bind_names = input_vec.clone();
        bind_names.remove(i);

        // For the first, restricted implementation
        // Only have T_? for parameters that are not required to be unbound
        let restricted_abstract: Vec<syn::Ident> = trait_params
            .clone()
            .into_iter()
            .enumerate()
            .filter(|&(x, _)| !out_vec.contains(&bind_names[x]))
            .map(|(_, e)| e)
            .collect();

        // Make sure  the above are unbound
        let restricted_trait: Vec<syn::Ident> = trait_params
            .clone()
            .into_iter()
            .enumerate()
            .map(|(x, e)| {
                if out_vec.contains(&bind_names[x]) {
                    unbound.clone()
                } else {
                    e
                }
            })
            .collect();

        let mut restricted_impl = restricted_trait.clone();
        restricted_impl.insert(i, unbound.clone());
        let mut restricted_type = restricted_trait.clone();
        restricted_type.insert(i, bound.clone());

        all_expanded.push(quote!{
            trait #trait_name<#(#trait_params: AbstractBind,)* B: Bindable, R: ProgramBindings, T: OutProgramBindings>{
                fn #bind_name(self, data : &B, program: &dyn Program, bindings: &mut R, out_bindings: &mut T) -> #context<#(#type_params),*>;
            }

            impl<#(#restricted_abstract: AbstractBind,)* B: Bindable, R: ProgramBindings, T: OutProgramBindings> #trait_name<#(#restricted_trait,)* B, R, T> for &#context<#(#restricted_impl),*> {
                fn #bind_name(self, data : &B, program: &dyn Program, bindings: &mut R, out_bindings: &mut T) -> #context<#(#restricted_type),*> {
                    Bindable::bind(
                        data,
                        program,
                        bindings,
                        out_bindings,
                        stringify!(#name).to_string(),
                    );
                    #context {
                        #(#fields : #restricted_type::new()),*
                    }
                }
            }

            impl<#(#trait_params: AbstractBind,)* B: Bindable, R: ProgramBindings, T: OutProgramBindings> #trait_name<#(#trait_params,)* B, R, T> for #context<#(#impl_params),*> {
                fn #bind_name(self, data : &B, program: &dyn Program, bindings: &mut R, out_bindings: &mut T) -> #context<#(#type_params),*>{
                    Bindable::bind(
                        data,
                        program,
                        bindings,
                        out_bindings,
                        stringify!(#name).to_string(),
                    );
                    #context {
                        #(#fields : #type_params::new()),*
                    }
                }
            }
        });
    }

    let mut collapsed_expanded = quote! {};
    for i in all_expanded.into_iter() {
        collapsed_expanded = quote! {
            #collapsed_expanded
            #i
        }
    }

    println!("{}", collapsed_expanded);

    // Hand the output tokens back to the compiler
    TokenStream::from(collapsed_expanded)
}
