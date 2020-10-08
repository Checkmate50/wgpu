extern crate proc_macro;
use itertools::Itertools;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Ident, Token};

use std::collections::HashSet;
use std::iter;

struct Parameters {
    params: HashSet<Ident>,
    outs: HashSet<Ident>,
}

impl Parse for Parameters {
    fn parse(input: ParseStream) -> Result<Self> {
        let vars = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        let mut params = Vec::new();
        let mut outs = Vec::new();
        for i in vars.into_iter() {
            if params.contains(&i) {
                outs.push(i)
            } else {
                params.push(i)
            }
        }
        Ok(Parameters {
            params: params.into_iter().collect(),
            outs: outs.into_iter().collect(),
        })
    }
}

/* struct name_field1_field2_field3 {}

impl name_field1_field2_field3 {
    fn bind_field1(&self) -> name_field2_field3{}
} */

/* [in ] field1
[in out] field2
[in out] field3
[in] field4

struct name<T1: AbstractBind, T2 : AbstractBind, T3: AbstractBind, T4: AbstractBind>{
    field1: T1,
    field2: T2,
    field3: T3,
    field4: T4,
}

impl name<T1: Unbound, T2: Unbound, T3: Unbound, T4: AbstractBind> {
    fn bind_field1(&self) -> name<Bound, T2, T3, T4>
}

impl name<T1: Unbound, T2: AbstractBind, T3: AbstractBind, T4:AbstractBind> {
    fn bind_field1(self) -> name<Bound, T2, T3>
} */

/*
impl name<T1: Unbound, T2: AbstractBind, T3: Bound> {
    fn bind_field1_consume(self) -> name<Bound, T2, T3>
}
*/
/* implementation 1
fn create_struct_name(mut set_of_params: Vec<Ident>) -> Ident {
    set_of_params.sort();
    let mut struct_name = format_ident!("ShaderContext");
    for i in set_of_params.into_iter() {
        struct_name = format_ident!("{}_{}", struct_name, i);
    }
    struct_name
}

fn create_bind_instructions_iter(
    set_of_params: Vec<Ident>,
) -> std::vec::IntoIter<(syn::Ident, syn::Ident)> {
    let mut result = Vec::new();
    for i in 0..set_of_params.len() {
        let mut subset_of_params = set_of_params.clone();
        let bind = subset_of_params.remove(i);
        result.push((bind, create_struct_name(subset_of_params)))
    }
    result.into_iter()
}

#[proc_macro]
pub fn my_macro(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as Parameters);

    let input_vec: Vec<Ident> = input.vars.into_iter().collect();

    let mut combined_types: Vec<Vec<Ident>> = Vec::new();

    for i in 0..input_vec.len() + 1 {
        let mut x: Vec<Vec<Ident>> = input_vec
            .clone()
            .into_iter()
            .combinations(i)
            .into_iter()
            .collect();
        combined_types.append(&mut x);
    }

    //println!("{:?}", combined_types);

    let mut all_expanded = Vec::new();

    for set_of_params in combined_types {
        let struct_name = create_struct_name(set_of_params.clone());
        let expanded = quote! {
            struct #struct_name {}
        };
        all_expanded.push(expanded);

        if set_of_params.len() == 0 {
            let implementations = quote! {
                impl #struct_name {
                    pub fn run(self) {
                        println!("hello")
                    }
                }
            };
            all_expanded.push(implementations);
        } else {
            for (bind, result_struct) in create_bind_instructions_iter(set_of_params) {
                let bind_name = format_ident!("bind_{}", bind);
                // todo when to use self and when to use &self
                let implementations = quote! {
                    impl #struct_name {
                        pub fn #bind_name(&self) -> #result_struct {
                            #result_struct {}
                        }
                    }
                };
                all_expanded.push(implementations);
            }
        }
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
} */

/* // Implementation 2
#[proc_macro]
pub fn generic_bindings(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as Parameters);

    let input_vec: Vec<Ident> = input.params.into_iter().collect();
    let out_vec: Vec<Ident> = input.outs.into_iter().collect();

    let mut all_expanded = Vec::new();

    all_expanded.push(quote! {
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

        trait AbstractContext {
            fn new() -> Self;
        }

        struct Context {}

        struct MutContext {}

        impl AbstractContext for Context {
            fn new() -> Self {
                Context {}
            }
        }

        impl AbstractContext for MutContext {
            fn new() -> Self {
                MutContext {}
            }
        }
    });

    let init: Vec<Ident> = iter::repeat(format_ident!("Unbound"))
        .take(input_vec.len())
        .collect();
    all_expanded.push(quote! {
        fn init() -> (#(#init),*, Context) {
            (#(#init{}),*, Context{})
        }
    });

    let run: Vec<Ident> = iter::repeat(format_ident!("Bound"))
        .take(input_vec.len())
        .collect();
    all_expanded.push(quote! {
        fn run<T : AbstractContext>(_: (#(#run),*, T)) {
            println!("hello")
        }
    });

    let variables: Vec<Ident> = (0..input_vec.len() - 1)
        .into_iter()
        .map(|x| format_ident!("param_{}", x))
        .collect();

    let bound = format_ident!("Bound");
    let unbound = format_ident!("Unbound");

    for i in 0..input_vec.len() {
        let bind_name = format_ident!("bind_{}", input_vec[i]);
        let bind_mutate = format_ident!("bind_{}_mutate", input_vec[i]);
        let bind_consume = format_ident!("bind_{}_consume", input_vec[i]);
        let mut type_params = variables.clone();
        type_params.insert(i, unbound.clone());
        let mut result_params = variables.clone();
        result_params.insert(i, bound.clone());
        if !out_vec.contains(&input_vec[i]) {
            all_expanded.push(quote! {
                fn #bind_name<#(#variables : AbstractBind),*>(_: &(#(#type_params),*, Context)) -> (#(#result_params),*, Context) {
                    (#(#result_params::new()),*, Context::new())
                }
            });
        } else {
            all_expanded.push(quote! {
                fn #bind_mutate<#(#variables : AbstractBind),*>(_: &(#(#type_params),*, Context)) -> (#(#result_params),*, MutContext) {
                    (#(#result_params::new()),*, MutContext::new())
                }
            });
        }
        all_expanded.push(quote! {
            fn #bind_consume<#(#variables : AbstractBind),*>(_: (#(#type_params),*, MutContext)) -> (#(#result_params),*, MutContext) {
                (#(#result_params::new()),*, MutContext::new())
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
} */

// Implementation 3
#[proc_macro]
pub fn generic_bindings(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as Parameters);

    let input_vec: Vec<Ident> = input.params.into_iter().collect();
    let out_vec: Vec<Ident> = input.outs.into_iter().collect();

    let mut all_expanded = Vec::new();

    all_expanded.push(quote! {
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
    });

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
    all_expanded.push(quote! {
        struct Context<#(#variables: AbstractBind),*> {
            #(#fields: #variables,)*
        }
        impl Context<#(#init),*> {
            fn new() -> Self {
                Context {
                    #(#fields: Unbound {},)*
                }
            }
        }
        impl Context<#(#run),*> {
            fn run(self) {
                println!("hello");
            }
        }
    });

    let bound = format_ident!("Bound");
    let unbound = format_ident!("Unbound");

    for i in 0..input_vec.len() {
        let trait_name = format_ident!("BindField{}", i + 1);
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

        println!("bind_names {:?}", bind_names);
        println!("out_vec {:?}", out_vec);

        // For the first, restricted implementation
        // Only have T_? for parameters that are not required to be unbound
        let restricted_abstract : Vec<syn::Ident> = trait_params.clone().into_iter().enumerate()
                .filter(|&(x, _)| !out_vec.contains(&bind_names[x]) )
                .map(|(_, e)| e).collect();

        println!("restricted_abstract {:?}", restricted_abstract);
        // Make sure  the above are unbound
        let restricted_trait : Vec<syn::Ident> = trait_params.clone().into_iter().enumerate()
                .map(|(x, e)| if out_vec.contains(&bind_names[x]) {unbound.clone()} else {e}).collect();

        let mut restricted_impl = restricted_trait.clone();
        restricted_impl.insert(i, unbound.clone());
        let mut restricted_type= restricted_trait.clone();
        restricted_type.insert(i, bound.clone());

        all_expanded.push(quote!{
            trait #trait_name<#(#trait_params: AbstractBind),*>{
                fn #bind_name(self) -> Context<#(#type_params),*>;
            }

            impl<#(#restricted_abstract: AbstractBind),*> #trait_name<#(#restricted_trait),*> for &Context<#(#restricted_impl),*> {
                fn #bind_name(self) -> Context<#(#restricted_type),*> {
                    Context {
                        #(#fields : #restricted_type::new()),*
                    }
                }
            }

            impl<#(#trait_params: AbstractBind),*> #trait_name<#(#trait_params),*> for Context<#(#impl_params),*> {
                fn #bind_name(self) -> Context<#(#type_params),*>{
                    Context {
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
