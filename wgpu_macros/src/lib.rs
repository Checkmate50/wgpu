#![feature(vec_remove_item)]
extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{braced, bracketed, parse_macro_input, Ident, Token};

use std::collections::HashMap;
use std::iter;

use rand::Rng;

#[derive(Debug, Clone)]
struct Parameters {
    group: Option<Ident>,
    quals: Vec<Ident>,
    glsl_type: Ident,
    name: Ident,
}

impl PartialEq for Parameters {
    fn eq(&self, other: &Parameters) -> bool {
        self.name == other.name
    }
}

impl Parse for Parameters {
    fn parse(input: ParseStream) -> Result<Self> {
        let qual_and_type;
        bracketed!(qual_and_type in input);
        let group = if qual_and_type.peek(Ident) {
            Some(qual_and_type.parse::<Ident>()?)
        } else {
            None
        };

        let qual_lst;
        bracketed!(qual_lst in qual_and_type);
        let mut quals = Vec::new();
        while !qual_lst.is_empty() {
            // loop and in tokens are not Ident's so they need to be handled differently
            if qual_lst.peek(Token!(loop)) {
                qual_lst.parse::<Token!(loop)>()?;
                quals.push(format_ident!("loop"));
            } else if qual_lst.peek(Token!(in)) {
                qual_lst.parse::<Token!(in)>()?;
                quals.push(format_ident!("in"));
            } else {
                quals.push(qual_lst.parse::<Ident>()?);
            }
        }

        let glsl_type = qual_and_type.parse::<Ident>()?;
        while !qual_and_type.is_empty() {
            let _x;
            bracketed!(_x in qual_and_type);
        }

        let name = input.parse::<Ident>()?;
        Ok(Parameters {
            group,
            glsl_type,
            quals: quals.into_iter().collect(),
            name,
        })
    }
}

struct Shader {
    params: Vec<Parameters>, //body: String,
}

impl Parse for Shader {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut params = Vec::new();
        while !input.peek(syn::token::Brace) {
            params.push(input.parse::<Parameters>()?);
            input.parse::<Token![;]>()?;
        }

        // The body/code of the shader
        let x;
        let y;
        braced!(x in input);
        braced!(y in x);
        if !y.is_empty() {
            y.step(|cursor| {
                (*cursor).token_stream();
                Ok(((), syn::buffer::Cursor::empty()))
            })?;
            //parse_any(&y)?;
        }
        Ok(Shader {
            params: params.into_iter().collect(),
        })
    }
}

struct Context {
    context: Ident,
    ins: Vec<Parameters>,
    outs: Vec<Parameters>,
}

impl Parse for Context {
    fn parse(input: ParseStream) -> Result<Self> {
        let context = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let shaders = Punctuated::<Shader, Token![,]>::parse_separated_nonempty(input)?;

        let mut ins = Vec::new();
        let mut outs = Vec::new();

        shaders.into_iter().for_each(|s| {
            println!();
            println!("ins : {:?}", ins);
            println!("outs : {:?}", outs);
            s.params.into_iter().for_each(|p| {
                if p.quals.contains(&format_ident!("in")) {
                    if outs.contains(&p) {
                        outs.remove_item(&p);
                    } else {
                        ins.push(p);
                    }
                } else if p.quals.contains(&format_ident!("out")) {
                    if ins.contains(&(p)) {
                        ins.remove_item(&p);
                    } else {
                        outs.push(p);
                    }
                }
            })
        });

        Ok(Context { context, ins, outs })
    }
}

// todo Maybe throw this in a file instead of using init()?
#[proc_macro]
pub fn init(_: TokenStream) -> TokenStream {
    TokenStream::from(quote! {
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

fn create_base_type(ty: &Ident) -> syn::GenericArgument {
    let mut data_type = syn::punctuated::Punctuated::new();
    match &ty.to_string()[..] {
        "vec3" => {
            //Vec<[f32; 3]>
            let mut array_type = syn::punctuated::Punctuated::new();
            array_type.push(syn::PathSegment {
                ident: format_ident!("f32"),
                arguments: syn::PathArguments::None,
            });

            let mut bracked_type = syn::punctuated::Punctuated::new();
            bracked_type.push(syn::GenericArgument::Type(syn::Type::Array(
                syn::TypeArray {
                    bracket_token: syn::token::Bracket(proc_macro2::Span::call_site()),
                    elem: Box::new(syn::Type::Path(syn::TypePath {
                        qself: None,
                        path: syn::Path {
                            leading_colon: None,
                            segments: array_type,
                        },
                    })),
                    semi_token: Token!(;)(proc_macro2::Span::call_site()),
                    len: syn::Expr::Lit(syn::ExprLit {
                        attrs: Vec::new(),
                        lit: syn::Lit::Int(syn::LitInt::new("3", proc_macro2::Span::call_site())),
                    }),
                },
            )));
            data_type.push(syn::PathSegment {
                ident: format_ident!("Vec"),
                arguments: syn::PathArguments::AngleBracketed(
                    syn::AngleBracketedGenericArguments {
                        args: bracked_type,
                        colon2_token: None,
                        lt_token: Token!(<)(proc_macro2::Span::call_site()),
                        gt_token: Token!(>)(proc_macro2::Span::call_site()),
                    },
                ),
            });
        }
        "mat4" => {
            //cgmath::Matrix4<f32>
            let mut inner_type = syn::punctuated::Punctuated::new();
            inner_type.push(syn::PathSegment {
                ident: format_ident!("f32"),
                arguments: syn::PathArguments::None,
            });
            let mut mat_type = syn::punctuated::Punctuated::new();
            mat_type.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path {
                    leading_colon: None,
                    segments: inner_type,
                },
            })));

            data_type.push(syn::PathSegment {
                ident: format_ident!("cgmath"),
                arguments: syn::PathArguments::None,
            });
            data_type.push(syn::PathSegment {
                ident: format_ident!("Matrix4"),
                arguments: syn::PathArguments::AngleBracketed(
                    syn::AngleBracketedGenericArguments {
                        args: mat_type,
                        colon2_token: None,
                        lt_token: Token!(<)(proc_macro2::Span::call_site()),
                        gt_token: Token!(>)(proc_macro2::Span::call_site()),
                    },
                ),
            });
        }

        _ => panic!(format!("Unsupported type for Vertex: {}", ty.to_string())),
    }

    syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: data_type,
        },
    }))
}

fn create_vertex(ty: &Ident) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    bind_ty.push(create_base_type(ty));

    let mut vertex_ty = syn::punctuated::Punctuated::new();
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("Vertex"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: bind_ty,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });

    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: vertex_ty,
        },
    })
}

fn create_bindgroup(ty: Vec<&Ident>) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    ty.iter().for_each(|t| bind_ty.push(create_base_type(t)));

    let mut vertex_ty = syn::punctuated::Punctuated::new();
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    vertex_ty.push(syn::PathSegment {
        ident: format_ident!("BindGroup{}", ty.len()),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: bind_ty,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });

    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: vertex_ty,
        },
    })
}

#[derive(Debug, Clone)]
enum ParamType {
    Vertex { num: u32, param: Parameters },
    Group { num: u32, param: Vec<Parameters> },
}

impl ParamType {
    fn get_params(&self) -> Vec<Parameters> {
        match self {
            ParamType::Vertex { num: _, param } => vec![param.clone()],
            ParamType::Group { num: _, param } => param.clone(),
        }
    }

    fn get_params_mut(&mut self) -> &mut Vec<Parameters> {
        match self {
            ParamType::Vertex { num: _, param: _ } => unreachable!(),
            ParamType::Group { num: _, param } => param,
        }
    }

    fn get_num(&self) -> u32 {
        match self {
            ParamType::Vertex { num, param: _ } => *num,
            ParamType::Group { num, param: _ } => *num,
        }
    }
}

fn process_params(params: Vec<Parameters>) -> Vec<ParamType> {
    let mut res = Vec::new();
    let mut group_map: HashMap<Ident, ParamType> = HashMap::new();
    let mut num_vertex = 0;
    let mut num_groups = 0;
    params.into_iter().for_each(|p| match p.group.clone() {
        Some(g) if group_map.contains_key(&g) => {
            group_map.get_mut(&g).unwrap().get_params_mut().push(p)
        }
        Some(g) => {
            group_map.insert(
                g,
                ParamType::Group {
                    num: num_groups,
                    param: vec![p],
                },
            );
            num_groups += 1
        }
        None => {
            res.push(ParamType::Vertex {
                num: num_vertex,
                param: p,
            });
            num_vertex += 1
        }
    });
    res.append(&mut group_map.into_iter().map(|(_, p)| p).collect());
    res
}

// Implementation 3
#[proc_macro]
pub fn generic_bindings(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let shader_params = parse_macro_input!(input as Context);

    // (name, type)
    let input_vec = shader_params.ins;

    let input_names: Vec<Ident> = input_vec.iter().map(|p| p.name.clone()).collect();
    let out_vec = shader_params.outs;
    let output_names: Vec<Ident> = out_vec.iter().map(|p| p.name.clone()).collect();

    let input_params = process_params(input_vec);

    println!("{:?}", input_params);

    let mut rng = rand::thread_rng();

    let n1: u8 = rng.gen();
    let context = format_ident!("Context{}", n1);

    let mut all_expanded = Vec::new();

    let variables: Vec<Ident> = (1..input_params.len() + 1)
        .into_iter()
        .map(|x| format_ident!("T{}", x))
        .collect();
    let fields: Vec<Ident> = (1..input_params.len() + 1)
        .into_iter()
        .map(|x| format_ident!("field{}", x))
        .collect();
    let init: Vec<Ident> = iter::repeat(format_ident!("Unbound"))
        .take(input_params.len())
        .collect();
    let run: Vec<Ident> = iter::repeat(format_ident!("Bound"))
        .take(input_params.len())
        .collect();
    let ctxloc = shader_params.context;
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
                vec![#(stringify!(#input_names).to_string()),*]
            }
        }

        impl #context<#(#run),*> {
            fn runnable<P, B>(&self, f: P) -> B where P: FnOnce() -> B{
                f()
            }
            fn can_pipe(&self, b : &dyn ContextInputs) {
                let a = vec![#(stringify!(#output_names).to_string()),*];
                assert!(b.inputs().iter().all(|item| a.contains(item)));
            }
        }

        let #ctxloc = #context::new();
    });

    let bound = format_ident!("Bound");
    let unbound = format_ident!("Unbound");

    let mut curr = -1;
    let index_numbers: Vec<syn::Expr> = iter::repeat_with(|| {
        curr += 1;
        syn::Expr::Lit(syn::ExprLit {
            attrs: Vec::new(),
            lit: syn::Lit::Int(syn::LitInt::new(
                &curr.to_string(),
                proc_macro2::Span::call_site(),
            )),
        })
    })
    .take(input_params.len())
    .collect();

    for i in 0..input_params.len() {
        let current_thing = input_params[i].clone();
        let trait_name = format_ident!("BindField{}{}", i + 1, n1);

        let bind_name = format_ident!(
            "{}",
            current_thing
                .get_params()
                .iter()
                .fold(format!("set"), |acc, p| format!("{}_{}", acc, p.name))
        );

        let is_vertex = true;
        let index = index_numbers.get(current_thing.get_num() as usize).unwrap();

        let data_type = match current_thing.clone() {
            ParamType::Vertex { param, .. } => create_vertex(&param.glsl_type),
            ParamType::Group { param, .. } => {
                create_bindgroup(param.iter().map(|p| &p.glsl_type).collect())
            }
        };

        let mut type_params = variables.clone();
        type_params.remove(i);
        type_params.insert(i, bound.clone());

        let mut trait_params = variables.clone();
        trait_params.remove(i);
        let mut impl_params = variables.clone();
        impl_params.remove(i);
        impl_params.insert(i, unbound.clone());

        // A copy of the input vec with the current param being bound removed so that the names match up with trait_params.
        let mut bind_names = input_params.clone();
        bind_names.remove(i);

        // For the first, restricted implementation
        // Only have T_? for parameters that are not required to be unbound
        let restricted_abstract: Vec<syn::Ident> = trait_params
            .clone()
            .into_iter()
            .enumerate()
            .filter(|&(x, _)| {
                !bind_names[x]
                    .get_params()
                    .iter()
                    .any(|p| out_vec.contains(p))
            })
            .map(|(_, e)| e)
            .collect();

        // Make sure the above are unbound
        let restricted_trait: Vec<syn::Ident> = trait_params
            .clone()
            .into_iter()
            .enumerate()
            .map(|(x, e)| {
                if bind_names[x]
                    .get_params()
                    .iter()
                    .any(|p| out_vec.contains(p))
                {
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

        match current_thing {
            ParamType::Vertex{..} => all_expanded.push(quote! {
                trait #trait_name<#(#trait_params: AbstractBind,)* >{
                    fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>;
                }

                impl<#(#trait_params: AbstractBind,)* > #trait_name<#(#trait_params,)*> for &#context<#(#impl_params),*> {
                    fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>{
                        rpass.set_vertex_buffer(#index as u32, data.get_buffer().slice(..));
                        #context {
                            #(#fields : #type_params::new()),*
                        }
                    }
                }
                impl<#(#trait_params: AbstractBind,)* > #trait_name<#(#trait_params,)*> for #context<#(#impl_params),*> {
                    fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>{
                        rpass.set_vertex_buffer(#index as u32, data.get_buffer().slice(..));
                        #context {
                            #(#fields : #type_params::new()),*
                        }
                    }
                }

            }),
            ParamType::Group{..} =>all_expanded.push(quote! {
            trait #trait_name<#(#trait_params: AbstractBind,)* >{
                fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>;
            }

            impl<#(#trait_params: AbstractBind,)* > #trait_name<#(#trait_params,)*> for &#context<#(#impl_params),*> {
                fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>{
                    rpass.set_bind_group(0 as u32, data.get_bind_group(), &[]);
                    #context {
                        #(#fields : #type_params::new()),*
                    }
                }
            }
            impl<#(#trait_params: AbstractBind,)* > #trait_name<#(#trait_params,)*> for #context<#(#impl_params),*> {
                fn #bind_name<'a>(self, rpass: &mut wgpu::RenderPass<'a>, data : &'a #data_type) -> #context<#(#type_params),*>{
                    rpass.set_bind_group(0 as u32, data.get_bind_group(), &[]);
                    #context {
                        #(#fields : #type_params::new()),*
                    }
                }
            }

        }),
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
}
