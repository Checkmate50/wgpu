#![allow(deprecated)]
extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, Token};

struct GetView {
    dimensions: syn::LitInt,
}

impl Parse for GetView {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(GetView {
            dimensions: input.parse()?,
        })
    }
}

fn create_path(n: u32, idx: u32, letters: &Vec<&str>) -> syn::Type {
    let mut bound_path = syn::punctuated::Punctuated::new();

    let mut multisample_path = syn::punctuated::Punctuated::new();
    multisample_path.push(syn::PathSegment {
        ident: format_ident!("MULTISAMPLE"),
        arguments: syn::PathArguments::None,
    });

    let mut sampletype_path = syn::punctuated::Punctuated::new();
    sampletype_path.push(syn::PathSegment {
        ident: format_ident!("SAMPLETYPE"),
        arguments: syn::PathArguments::None,
    });

    let mut viewdimension_path = syn::punctuated::Punctuated::new();
    viewdimension_path.push(syn::PathSegment {
        ident: format_ident!("VIEWDIMENSION"),
        arguments: syn::PathArguments::None,
    });

    let mut bracked_type = syn::punctuated::Punctuated::new();
    bracked_type.push(syn::GenericArgument::Lifetime(syn::Lifetime::new(
        "'a",
        proc_macro2::Span::call_site(),
    )));
    bracked_type.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: multisample_path,
        },
    })));
    bracked_type.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: sampletype_path,
        },
    })));
    bracked_type.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: viewdimension_path,
        },
    })));

    let mut bind_group_generic_types = syn::punctuated::Punctuated::new();

    for i in 0..n {
        if i == idx {
            let mut texture_type: syn::punctuated::Punctuated<
                syn::PathSegment,
                syn::token::Colon2,
            > = syn::punctuated::Punctuated::new();
            texture_type.push(syn::PathSegment {
                ident: format_ident!("TextureData"),
                arguments: syn::PathArguments::AngleBracketed(
                    syn::AngleBracketedGenericArguments {
                        args: bracked_type.clone(),
                        colon2_token: None,
                        lt_token: Token!(<)(proc_macro2::Span::call_site()),
                        gt_token: Token!(>)(proc_macro2::Span::call_site()),
                    },
                ),
            });
            bind_group_generic_types.push(syn::GenericArgument::Type(syn::Type::Path(
                syn::TypePath {
                    qself: None,
                    path: syn::Path {
                        leading_colon: None,
                        segments: texture_type,
                    },
                },
            )));
        }

        if i < letters.len() as u32 {
            let mut letter_type = syn::punctuated::Punctuated::new();
            letter_type.push(syn::PathSegment {
                ident: format_ident!("{}", letters[i as usize]),
                arguments: syn::PathArguments::None,
            });
            bind_group_generic_types.push(syn::GenericArgument::Type(syn::Type::Path(
                syn::TypePath {
                    qself: None,
                    path: syn::Path {
                        leading_colon: None,
                        segments: letter_type,
                    },
                },
            )));
        }
    }

    bound_path.push(syn::PathSegment {
        ident: format_ident!("BindGroup{}", n),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: bind_group_generic_types,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: bound_path,
        },
    })
}

pub fn sub_module_get_view_func(input: TokenStream) -> TokenStream {
    let get_view_params = parse_macro_input!(input as GetView);

    let idx = get_view_params.dimensions.base10_parse::<u32>().unwrap();

    let mut all_expanded = Vec::new();
    let other_letters: Vec<&str> = vec![
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P",
    ]
    .into_iter()
    .take((idx - 1) as usize)
    .collect();

    for i in 0..idx {
        //BindGroup3<TextureData<'a, MULTISAMPLE, SAMPLETYPE, VIEWDIMENSION>, C, D>
        let struct_type = create_path(idx, i, &other_letters);
        let function_name = format_ident!("get_view_{}", i);

        let index = i as usize;
        let generic_letters: Vec<syn::Ident> = other_letters
            .iter()
            .map(|x| format_ident!("{}", x))
            .collect();

        all_expanded.push(quote! {
            impl<'a, const MULTISAMPLE: TextureMultisampled, const SAMPLETYPE: wgpu::TextureSampleType,
        const VIEWDIMENSION: wgpu::TextureViewDimension, #(#generic_letters : WgpuType),*>
        #struct_type
    {
        pub fn #function_name(&self, desc: &wgpu::TextureViewDescriptor) -> wgpu::TextureView {
            match self.data.get(#index).unwrap() {
                BoundData::Texture { data, .. } => data.create_view(desc),
                _ => panic!("not a texture"),
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
