use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, Ident, LitStr, Token};

use std::collections::HashMap;
use std::iter;
use zerocopy::AsBytes as _;

use rand::Rng;

// This is a parameter which will need to be bound to.
// A parameter either has a group or it doesn't
//
#[derive(Debug, Clone)]
struct Parameters {
    location: u32,
    group: Option<u32>,
    glsl_type: naga::Type,
    name: syn::Ident,
}

impl PartialEq for Parameters {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for Parameters {}

enum EntryFunction {
    Vertex(LitStr),
    Fragment(LitStr),
    Compute(LitStr),
}

impl Parse for EntryFunction {
    fn parse(input: ParseStream) -> Result<Self> {
        let entry = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let name = input.parse::<LitStr>()?;
        input.parse::<Token![,]>()?;
        match entry.to_string().as_str() {
            "vertex" => Ok(EntryFunction::Vertex(name)),
            "fragment" => Ok(EntryFunction::Fragment(name)),
            "compute" => Ok(EntryFunction::Compute(name)),
            _ => panic!("not a valid entry point"),
        }
    }
}

fn is_entry_function(e: &naga::EntryPoint, vec_f: &Vec<EntryFunction>) -> bool {
    vec_f.iter().any(|x| match x {
        EntryFunction::Vertex(n) => n.value() == e.name,
        EntryFunction::Fragment(n) => n.value() == e.name,
        EntryFunction::Compute(n) => n.value() == e.name,
    })
}

// Contains the 'params'(Parameters) of one or more shaders which make up a pipeline context which will be created for the user at `context`
// This also contains the full naga `module` for the shader provided along with it's verification `info`
struct Context {
    context: Ident,
    params: Vec<Parameters>,
    module: naga::Module,
    info: naga::valid::ModuleInfo,
}

impl Parse for Context {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut entry_functions = Vec::new();
        // find entry points
        while !input.peek(Token![let]) {
            entry_functions.push(input.parse::<EntryFunction>()?);
        }

        // Ready to parse the shader
        input.parse::<Token![let]>()?;
        let context = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;

        let module = naga::front::wgsl::parse_str(&input.to_string()).unwrap();

        // We converted the rest of the parse into a stream so just chew up the rest
        while !input.is_empty() {
            input.step(|cursor| {
                let rest = *cursor;
                while let Some((_, next)) = rest.token_tree() {
                    return Ok(((), next));
                }
                unreachable!()
            })?;
        }

        // Validate the nage module
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let mut params: Vec<Parameters> = Vec::new();
        module.entry_points.iter().for_each(|f| {
            println!("{:?}", f.name);
            if is_entry_function(f, &entry_functions) {
                f.function.arguments.iter().for_each(|arg| {
                    match (arg.name.as_ref(), arg.binding.as_ref()) {
                        //https://docs.rs/naga/0.4.2/naga/enum.BuiltIn.html Most builtin's are just provided by the runtime. Some of these like position or baseVertex are probably something I should look at todo
                        (Some(name), Some(naga::Binding::BuiltIn(_))) => {
                            println!("Builtin {} {:?}", name, module.types.try_get(arg.ty))
                        }
                        (Some(name), Some(naga::Binding::Location { location, .. })) => {
                            println!(
                                "Location {} {:?} {:?}",
                                name,
                                location,
                                module.types.try_get(arg.ty)
                            );
                            params.push(Parameters {
                                location: *location,
                                group: None,
                                glsl_type: module.types.try_get(arg.ty).unwrap().clone(),
                                name: quote::format_ident!("{}", name),
                            })
                        }
                        //todo I'm not sure when this case fires and would like to know
                        (None, _) => {
                            println!("idk why there is no name")
                        }
                        //todo I wondered when None would get triggered. For example, if a fragment shader takes in a struct, that counts as a None Binding.
                        (_, None) => {
                            println!("hi")
                        }
                    }
                })
            }
        });
        module
            .global_variables
            .iter()
            // Filtering for uniform variables in bindgroups and handles to variable in bindgroups(like textures/samplers)
            // todo should this also use StorageClass::Storage??
            .filter(|var| {
                var.1.class == naga::StorageClass::Uniform
                    || var.1.class == naga::StorageClass::Handle
            })
            .for_each(|var| {
                println!(
                    "{} {:?} {:?} {:?}",
                    var.1.name.as_ref().unwrap(),
                    module.types.try_get(var.1.ty),
                    var.1.binding.as_ref().unwrap(),
                    var.1.class,
                );
                params.push(Parameters {
                    location: var.1.binding.as_ref().unwrap().binding,
                    group: Some(var.1.binding.as_ref().unwrap().group),
                    glsl_type: module.types.try_get(var.1.ty).unwrap().clone(),
                    name: quote::format_ident!("{}", var.1.name.as_ref().unwrap()),
                })
            });
        Ok(Context {
            context,
            params,
            module,
            info,
        })
    }
}

// Convert the naga VectorSize to it's corresponding number as a string
fn numerize(dim: &naga::VectorSize) -> &str {
    match dim {
        naga::VectorSize::Bi => "2",
        naga::VectorSize::Tri => "3",
        naga::VectorSize::Quad => "4",
    }
}

// Convert u8 to u32
fn byteify(width: &u8) -> u32 {
    *width as u32 * 8
}

// create the cgmath::Matrix type. This is assumed to be a square matrix with dimension `dim` and type `f<width>`
fn create_mat_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    dim: &naga::VectorSize,
    width: &u8,
    // maybe other args with other mat types
) {
    let mut inner_type = syn::punctuated::Punctuated::new();
    inner_type.push(syn::PathSegment {
        ident: format_ident!("f{}", byteify(width)),
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
        ident: format_ident!("Matrix{}", numerize(dim)),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: mat_type,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_vec_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    vec_type: syn::Ident,
) {
    let mut array_type = syn::punctuated::Punctuated::new();
    array_type.push(syn::PathSegment {
        ident: vec_type,
        arguments: syn::PathArguments::None,
    });

    let mut bracked_type = syn::punctuated::Punctuated::new();
    bracked_type.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: array_type,
        },
    })));
    data_type.push(syn::PathSegment {
        ident: format_ident!("Vec"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: bracked_type,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_vec_array_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    vec_type: syn::Ident,
    n: &naga::VectorSize,
) {
    let mut array_type = syn::punctuated::Punctuated::new();
    array_type.push(syn::PathSegment {
        ident: vec_type,
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
                lit: syn::Lit::Int(syn::LitInt::new(
                    &numerize(n),
                    proc_macro2::Span::call_site(),
                )),
            }),
        },
    )));
    data_type.push(syn::PathSegment {
        ident: format_ident!("Vec"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: bracked_type,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_buffer_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    generic: syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    is_buffer: bool,
) {
    let mut buffer_binding_type_path = syn::punctuated::Punctuated::new();
    buffer_binding_type_path.push(syn::PathSegment {
        ident: format_ident!("wgpu"),
        arguments: syn::PathArguments::None,
    });
    buffer_binding_type_path.push(syn::PathSegment {
        ident: format_ident!("BufferBindingType"),
        arguments: syn::PathArguments::None,
    });
    if is_buffer {
        buffer_binding_type_path.push(syn::PathSegment {
            ident: format_ident!("Storage"),
            arguments: syn::PathArguments::None,
        });
    } else {
        buffer_binding_type_path.push(syn::PathSegment {
            ident: format_ident!("Uniform"),
            arguments: syn::PathArguments::None,
        });
    }
    let mut buffer_binding_type_field = syn::punctuated::Punctuated::new();
    if is_buffer {
        buffer_binding_type_field.push(syn::FieldValue {
            attrs: Vec::new(),
            member: syn::Member::Named(format_ident!("read_only")),
            colon_token: Some(Token!(:)(proc_macro2::Span::call_site())),
            expr: syn::Expr::Lit(syn::ExprLit {
                attrs: Vec::new(),
                lit: syn::Lit::Bool(syn::LitBool {
                    value: false,
                    span: proc_macro2::Span::call_site(),
                }),
            }),
        });
    }

    let mut generic_args = syn::punctuated::Punctuated::new();
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Struct(
        syn::ExprStruct {
            attrs: Vec::new(),
            path: syn::Path {
                leading_colon: None,
                segments: buffer_binding_type_path,
            },
            brace_token: syn::token::Brace(proc_macro2::Span::call_site()),
            fields: buffer_binding_type_field,
            dot2_token: None,
            rest: None,
        },
    )));
    generic_args.push(syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: generic,
        },
    })));

    data_type.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    data_type.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    data_type.push(syn::PathSegment {
        ident: format_ident!("BufferData"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            colon2_token: Some(Token!(::)(proc_macro2::Span::call_site())),
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            args: generic_args,
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_sampler_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    comparison: &bool,
) {
    data_type.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    data_type.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    let mut comparable_path = syn::punctuated::Punctuated::new();
    comparable_path.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    comparable_path.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    comparable_path.push(syn::PathSegment {
        ident: format_ident!("SamplerComparison"),
        arguments: syn::PathArguments::None,
    });
    comparable_path.push(syn::PathSegment {
        ident: if *comparison {
            format_ident!("True")
        } else {
            format_ident!("False")
        },
        arguments: syn::PathArguments::None,
    });
    let mut filterable_path = syn::punctuated::Punctuated::new();
    filterable_path.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    filterable_path.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    filterable_path.push(syn::PathSegment {
        ident: format_ident!("SamplerFiltering"),
        arguments: syn::PathArguments::None,
    });
    //todo push True if filterable
    filterable_path.push(syn::PathSegment {
        ident: if false {
            format_ident!("True")
        } else {
            format_ident!("False")
        },
        arguments: syn::PathArguments::None,
    });
    let mut generic_args = syn::punctuated::Punctuated::new();
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Path(
        syn::ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: syn::Path {
                leading_colon: None,
                segments: comparable_path,
            },
        },
    )));
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Path(
        syn::ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: syn::Path {
                leading_colon: None,
                segments: filterable_path,
            },
        },
    )));

    data_type.push(syn::PathSegment {
        ident: format_ident!("SamplerData"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            args: generic_args,
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_texture_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    dim: &naga::ImageDimension,
    arrayed: &bool,
    class: &naga::ImageClass,
) {
    let mut multi_sample_path = syn::punctuated::Punctuated::new();
    multi_sample_path.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    multi_sample_path.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    multi_sample_path.push(syn::PathSegment {
        ident: format_ident!("TextureMultisampled"),
        arguments: syn::PathArguments::None,
    });
    //todo not sure what kind is here for in Multisampled
    multi_sample_path.push(syn::PathSegment {
        ident: if let naga::ImageClass::Sampled {
            multi: true,
            kind: _,
        } = class
        {
            format_ident!("True")
        } else {
            format_ident!("False")
        },
        arguments: syn::PathArguments::None,
    });

    let mut view_dimension_path = syn::punctuated::Punctuated::new();
    view_dimension_path.push(syn::PathSegment {
        ident: format_ident!("wgpu"),
        arguments: syn::PathArguments::None,
    });
    view_dimension_path.push(syn::PathSegment {
        ident: format_ident!("TextureViewDimension"),
        arguments: syn::PathArguments::None,
    });
    view_dimension_path.push(syn::PathSegment {
        ident: match dim {
            naga::ImageDimension::D1 => format_ident!("D1"),
            naga::ImageDimension::D2 => {
                if *arrayed {
                    format_ident!("D2Array")
                } else {
                    format_ident!("D2")
                }
            }
            naga::ImageDimension::D3 => format_ident!("D3"),
            naga::ImageDimension::Cube => {
                if *arrayed {
                    format_ident!("CubeArray")
                } else {
                    format_ident!("Cube")
                }
            }
        },
        arguments: syn::PathArguments::None,
    });

    let mut sample_type_path = syn::punctuated::Punctuated::new();
    sample_type_path.push(syn::PathSegment {
        ident: format_ident!("wgpu"),
        arguments: syn::PathArguments::None,
    });
    sample_type_path.push(syn::PathSegment {
        ident: format_ident!("TextureSampleType"),
        arguments: syn::PathArguments::None,
    });

    let mut sample_type_field = syn::punctuated::Punctuated::new();

    if let naga::ImageClass::Sampled { kind, multi: _ } = class {
        match kind {
            naga::ScalarKind::Sint => {
                sample_type_path.push(syn::PathSegment {
                    ident: format_ident!("Sint"),
                    arguments: syn::PathArguments::None,
                });
            }
            naga::ScalarKind::Uint => {
                sample_type_path.push(syn::PathSegment {
                    ident: format_ident!("Uint"),
                    arguments: syn::PathArguments::None,
                });
            }
            naga::ScalarKind::Bool => {
                unimplemented!() // Yeah, there is no corresponding thing for bool so idk
            }
            naga::ScalarKind::Float => {
                sample_type_path.push(syn::PathSegment {
                    ident: format_ident!("Float"),
                    arguments: syn::PathArguments::None,
                });

                sample_type_field.push(syn::FieldValue {
                    attrs: Vec::new(),
                    member: syn::Member::Named(format_ident!("filterable")),
                    colon_token: Some(Token!(:)(proc_macro2::Span::call_site())),
                    expr: syn::Expr::Lit(syn::ExprLit {
                        attrs: Vec::new(),
                        lit: syn::Lit::Bool(syn::LitBool {
                            value: false,
                            span: proc_macro2::Span::call_site(),
                        }),
                    }),
                });
            }
        }
    } else if let naga::ImageClass::Depth = class {
        sample_type_path.push(syn::PathSegment {
            ident: format_ident!("Depth"),
            arguments: syn::PathArguments::None,
        });
    }
    let mut generic_args = syn::punctuated::Punctuated::new();
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Path(
        syn::ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: syn::Path {
                leading_colon: None,
                segments: multi_sample_path,
            },
        },
    )));
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Struct(
        syn::ExprStruct {
            attrs: Vec::new(),
            path: syn::Path {
                leading_colon: None,
                segments: sample_type_path,
            },
            brace_token: syn::token::Brace(proc_macro2::Span::call_site()),
            fields: sample_type_field,
            dot2_token: None,
            rest: None,
        },
    )));
    generic_args.push(syn::GenericArgument::Const(syn::Expr::Path(
        syn::ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: syn::Path {
                leading_colon: None,
                segments: view_dimension_path,
            },
        },
    )));

    data_type.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    data_type.push(syn::PathSegment {
        ident: format_ident!("bind"),
        arguments: syn::PathArguments::None,
    });
    data_type.push(syn::PathSegment {
        ident: format_ident!("TextureData"),
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            args: generic_args,
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn letterize(kind: &naga::ScalarKind) -> &str {
    match kind {
        naga::ScalarKind::Float => "f",
        naga::ScalarKind::Uint => "u",
        naga::ScalarKind::Sint => "i",
        naga::ScalarKind::Bool => {
            unimplemented!()
        }
    }
}

// Like Vec<[f32;4]>
fn create_new_base_type(
    ty: &naga::Type,
) -> syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2> {
    match ty {
        // For now, I am going to assume 32 bit width
        naga::Type {
            name: _,
            inner: naga::TypeInner::Scalar { kind, width },
        } => {
            //Vec<f32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            //todo change this based on location vs uniform
            create_vec_type(
                &mut generic_type,
                format_ident!("{}{}", letterize(kind), byteify(width)),
            );
            generic_type
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Vector { size, kind, width },
        } => {
            //Vec<[f32; dim]>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_vec_array_type(
                &mut generic_type,
                format_ident!("{}{}", letterize(kind), byteify(width)),
                size,
            );
            generic_type
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Array {
                base: (),
                size: (),
                stride: (),
            },
        } => {
           // [f32; size]
            unimplemented!()
        }
        naga::Type {
            name: _,
            inner:
                naga::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                },
        } if columns == rows => {
            //cgmath::Matrix4<f32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_mat_type(&mut generic_type, columns, width);
            generic_type
        }
        naga::Type {
            name: Some(name), // the name of the struct
            inner:
                naga::TypeInner::Struct {
                    top_level: _, // Whether this is the root struct or an inner struct
                    members: _,
                    span: _,
                },
        } => {
            //Struct<const BINDINGTYPE: wgpu::BufferBindingType>
            let mut buffer_binding_type_path = syn::punctuated::Punctuated::new();
            buffer_binding_type_path.push(syn::PathSegment {
                ident: format_ident!("wgpu"),
                arguments: syn::PathArguments::None,
            });
            buffer_binding_type_path.push(syn::PathSegment {
                ident: format_ident!("BufferBindingType"),
                arguments: syn::PathArguments::None,
            });
            if false {
                // todo should be is_buffer
                buffer_binding_type_path.push(syn::PathSegment {
                    ident: format_ident!("Storage"),
                    arguments: syn::PathArguments::None,
                });
            } else {
                buffer_binding_type_path.push(syn::PathSegment {
                    ident: format_ident!("Uniform"),
                    arguments: syn::PathArguments::None,
                });
            }
            let mut generic_type = syn::punctuated::Punctuated::new();
            generic_type.push(syn::GenericArgument::Const(syn::Expr::Struct(
                syn::ExprStruct {
                    attrs: Vec::new(),
                    path: syn::Path {
                        leading_colon: None,
                        segments: buffer_binding_type_path,
                    },
                    brace_token: syn::token::Brace(proc_macro2::Span::call_site()),
                    fields: syn::punctuated::Punctuated::new(),
                    dot2_token: None,
                    rest: None,
                },
            )));

            let mut struct_type = syn::punctuated::Punctuated::new();
            struct_type.push(syn::PathSegment {
                ident: format_ident!("{}", name),
                arguments: syn::PathArguments::AngleBracketed(
                    syn::AngleBracketedGenericArguments {
                        args: generic_type,
                        colon2_token: None,
                        lt_token: Token!(<)(proc_macro2::Span::call_site()),
                        gt_token: Token!(>)(proc_macro2::Span::call_site()),
                    },
                ),
            });
            struct_type
        }
        _ => panic!("Unsupported new base type {:?}", ty),
    }
}

// Like BufferData<Vec<[f32;4]>>
fn create_type_with_wrapper(ty: &naga::Type) -> syn::GenericArgument {
    let mut data_type = syn::punctuated::Punctuated::new();
    match ty {
        naga::Type {
            name: _,
            inner:
                naga::TypeInner::Scalar { .. }
                | naga::TypeInner::Vector { .. }
                | naga::TypeInner::Matrix { .. },
        } => {
            create_buffer_type(&mut data_type, create_new_base_type(ty), true);
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Struct { .. },
        } => {
            data_type = create_new_base_type(ty);
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Sampler { comparison },
        } => create_sampler_type(&mut data_type, comparison),
        naga::Type {
            name: _,
            inner:
                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                },
        } => {
            create_texture_type(&mut data_type, dim, arrayed, class);
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. },
        } => {
            panic!("Pointers are unsupported types")
        }
        naga::Type {
            name: _,
            inner: naga::TypeInner::Array { .. },
        } => {
            panic!("I am unsure if arrays should be supported")
        }
    }
    syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: data_type,
        },
    }))
}

fn create_vertex(ty: &naga::Type) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    bind_ty.push(create_type_with_wrapper(ty));

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
            colon2_token: Some(Token!(::)(proc_macro2::Span::call_site())),
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

fn create_bindgroup(ty: Vec<&naga::Type>) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    ty.iter()
        .for_each(|t| bind_ty.push(create_type_with_wrapper(t)));

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
            colon2_token: Some(Token!(::)(proc_macro2::Span::call_site())),
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

//todo This should create a new group if there is no group name.
// Only create a vertex when there is the vertex qualifier
fn process_params(params: Vec<Parameters>) -> Vec<ParamType> {
    let mut res = Vec::new();
    let mut group_map: HashMap<u32, ParamType> = HashMap::new();
    let mut num_vertex = 0;
    params.into_iter().for_each(|p| match p.group.clone() {
        Some(g) if group_map.contains_key(&g) => {
            group_map.get_mut(&g).unwrap().get_params_mut().push(p)
        }
        Some(num) => {
            group_map.insert(
                num,
                ParamType::Group {
                    num,
                    param: vec![p],
                },
            );
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

fn make_trait(t: String) -> syn::Type {
    let mut bound_path = syn::punctuated::Punctuated::new();
    bound_path.push(syn::PathSegment {
        ident: format_ident!("{}", t),
        arguments: syn::PathArguments::None,
    });
    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: bound_path,
        },
    })
}

fn bound() -> syn::Type {
    let mut bound_path = syn::punctuated::Punctuated::new();
    bound_path.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    bound_path.push(syn::PathSegment {
        ident: format_ident!("Bound"),
        arguments: syn::PathArguments::None,
    });
    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: bound_path,
        },
    })
}

fn unbound() -> syn::Type {
    let mut bound_path = syn::punctuated::Punctuated::new();
    bound_path.push(syn::PathSegment {
        ident: format_ident!("pipeline"),
        arguments: syn::PathArguments::None,
    });
    bound_path.push(syn::PathSegment {
        ident: format_ident!("Unbound"),
        arguments: syn::PathArguments::None,
    });
    syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: bound_path,
        },
    })
}

// Implementation 3
pub fn sub_module_generic_bindings(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let shader_params = parse_macro_input!(input as Context);
    let spirv = naga::back::spv::write_vec(
        &shader_params.module,
        &shader_params.info,
        &naga::back::spv::Options::default(),
    )
    .unwrap();
    let spirv_bytes = spirv.as_bytes();

    // (name, type)
    let param_vec = shader_params.params.clone();

    let params = process_params(param_vec);

    let mut rng = rand::thread_rng();

    let n1: u8 = rng.gen();
    let context = format_ident!("Context{}", n1);

    let mut all_expanded = Vec::new();

    println!("{:?}", params);

    // Setting up struct
    //let struct_set = std::collections::HashSet::new();
    shader_params.module.types.iter().for_each(|(_,t)| {
        if let naga::Type {
            name: Some(name),
            inner:
                naga::TypeInner::Struct {
                    top_level:_,
                    span: _,
                    mut members,
                },
        } = t.clone()
        {
            members.sort_by(|a, b| a.offset.cmp(&b.offset));
            let (mut field_name, mut field_type) = (Vec::new(), Vec::new());
            members.iter().for_each(|m| {
                field_name.push(format_ident!("{}", m.name.as_ref().unwrap()));
                field_type.push(create_new_base_type(
                    shader_params.module.types.try_get(m.ty).unwrap(),
                ))
            });
            let struct_name = format_ident!("{}", name);
            all_expanded.push(quote! {
                struct #struct_name<const BINDINGTYPE: wgpu::BufferBindingType> {
                    #(#field_name: #field_type,)*
                }

                impl<const BINDINGTYPE: wgpu::BufferBindingType> pipeline::bind::WgpuType for #struct_name<BINDINGTYPE> {
                    fn bind(
                        &self,
                        device: &wgpu::Device,
                        qual: Option<pipeline::shared::QUALIFIER>,
                    ) -> pipeline::bind::BoundData {
                        use pipeline::align::Alignment;
                        pipeline::bind::BoundData::new_buffer(
                            device,
                            &self.align_bytes(),
                            1 as u64, //todo this might not be correct
                            Self::size_of(),
                            qual,
                            Self::create_binding_type(),
                        )
                    }
                    fn size_of() -> usize {
                        use pipeline::align::Alignment;
                        <#struct_name<BINDINGTYPE>>::alignment_size()
                    }
                    fn create_binding_type() -> wgpu::BindingType {
                        wgpu::BindingType::Buffer {
                            ty: BINDINGTYPE,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(Self::size_of() as u64),
                        }
                    }
                    fn get_qualifiers() -> Option<pipeline::shared::QUALIFIER> {
                        match BINDINGTYPE {
                            wgpu::BufferBindingType::Uniform => Some(pipeline::shared::QUALIFIER::UNIFORM),
                            wgpu::BufferBindingType::Storage { read_only: _ } => {
                                Some(pipeline::shared::QUALIFIER::BUFFER)
                            }
                        }
                    }
                }

                impl<const BINDINGTYPE: wgpu::BufferBindingType> pipeline::align::Alignment for #struct_name<BINDINGTYPE> {
                    fn alignment_size() -> usize {
                        use pipeline::align::Alignment;
                        #(<#field_type>::alignment_size())+*
                    }
                    fn align_bytes(&self) -> Vec<u8> {
                        [#(self.#field_name.align_bytes(),)*].concat()
                    }
                }

            });
        }
    });

    // Setting up context
    let variables: Vec<syn::Type> = (1..params.len() + 1)
        .into_iter()
        .map(|x| make_trait(format!("T{}", x)))
        .collect();
    let fields: Vec<Ident> = (1..params.len() + 1)
        .into_iter()
        .map(|x| format_ident!("field{}", x))
        .collect();
    let init: Vec<syn::Type> = iter::repeat(unbound()).take(params.len()).collect();
    let run: Vec<syn::Type> = iter::repeat(bound()).take(params.len()).collect();
    let ctxloc = shader_params.context;

    // For setting up pipeline
    let mut bind_group_types: Vec<(u32, syn::Type)> = Vec::new();
    let mut vertex_types: Vec<(u32, syn::Type)> = Vec::new();

    params.clone().into_iter().for_each(|a| match a {
        ParamType::Vertex { num, param } => {
            vertex_types.push((num, create_vertex(&param.glsl_type)))
        }
        ParamType::Group { num, param } => bind_group_types.push((
            num,
            create_bindgroup(param.iter().map(|p| &p.glsl_type).collect()),
        )),
    });

    bind_group_types.sort_by(|(a, _), (b, _)| a.cmp(b));
    vertex_types.sort_by(|(a, _), (b, _)| a.cmp(b));

    let sorted_bind_group_types: Vec<syn::Type> =
        bind_group_types.into_iter().map(|(_, x)| x).collect();

    let sorted_vertex_types: Vec<syn::Type> = vertex_types.into_iter().map(|(_, x)| x).collect();

    all_expanded.push(quote! {
        struct #context<'a,  T : pipeline :: RuntimePass<'a>, #(#variables: pipeline::AbstractBind),*> {
            phantom: std::marker::PhantomData<&'a T>,
            #(#fields: #variables,)*
        }

        impl <'a,  T : pipeline :: RuntimePass<'a>> #context<'a, T, #(#init),*> {
            fn new() -> Self {
                #context {
                    phantom: std::marker::PhantomData,
                    #(#fields: pipeline::Unbound {},)*
                }
            }
            fn get_bindgroup_layout(&self, device : &wgpu::Device) -> Vec<wgpu::BindGroupLayout> {
                vec![#(#sorted_bind_group_types::get_layout(device),)*]
            }
            fn get_vertex_sizes(&self) -> Vec<usize> {
                vec![#(#sorted_vertex_types::size_of(),)*]
            }
            fn get_entry_points(&self) -> Vec<String> {
                //todo actual entry points
                vec!["vs_main".to_string(), "fs_main".to_string()]
            }
            fn get_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::util::make_spirv(&[#(#spirv_bytes),*]),
                    flags: wgpu::ShaderFlags::VALIDATION,
                })
            }
        }

        impl <'a,  T : pipeline :: RuntimePass<'a>> #context<'a, T, #(#run),*> {
            fn runnable<P>(&self, t: &mut T, f: P)
        where
            P: FnOnce(&mut T),
        {
            f(t)
        }
        }

        let #ctxloc = #context::new();
    });

    let bound = bound();
    let unbound = unbound();

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
    .take(params.len())
    .collect();

    for i in 0..params.len() {
        let current_thing = params[i].clone();
        let trait_name = format_ident!("BindField{}{}", i + 1, n1);

        let bind_name = format_ident!(
            "{}",
            current_thing
                .get_params()
                .iter()
                .fold("set".to_string(), |acc, p| format!("{}_{}", acc, p.name))
        );

        let index = index_numbers.get(current_thing.get_num() as usize).unwrap();

        let data_type = match current_thing.clone() {
            ParamType::Vertex { param, .. } => create_vertex(&param.glsl_type),
            ParamType::Group { param, .. } => {
                create_bindgroup(param.iter().map(|p| (&p.glsl_type)).collect())
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
        let mut bind_names = params.clone();
        bind_names.remove(i);

        match current_thing {
            ParamType::Vertex{..} => all_expanded.push(quote! {
                trait #trait_name<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* >{
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>;
                }

                impl<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#trait_params,)*> for &#context<'a, T, #(#impl_params),*> {
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>{
                        rpass.set_vertex_buffer(#index as u32, data.get_buffer().slice(..));
                        #context {
                            phantom: std::marker::PhantomData,
                            #(#fields : #type_params::new()),*
                        }
                    }
                }

                impl<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#trait_params,)*> for #context<'a, T, #(#impl_params),*> {
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>{
                        rpass.set_vertex_buffer(#index as u32, data.get_buffer().slice(..));
                        #context {
                            phantom: std::marker::PhantomData,
                            #(#fields : #type_params::new()),*
                        }
                    }
                }
            }),
            ParamType::Group{..} =>all_expanded.push(quote! {
                trait #trait_name<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* >{
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>;
                }

                impl<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#trait_params,)*> for &#context<'a, T, #(#impl_params),*> {
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>{
                        rpass.set_bind_group(#index as u32, data.get_bind_group(), &[]);
                        #context {
                            phantom: std::marker::PhantomData,
                            #(#fields : #type_params::new()),*
                        }
                    }
                }

                impl<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#trait_params,)*> for #context<'a, T, #(#impl_params),*> {
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>{
                        rpass.set_bind_group(#index as u32, data.get_bind_group(), &[]);
                        #context {
                            phantom: std::marker::PhantomData,
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
