use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::{braced, bracketed, parse_macro_input, Ident, Token};

use std::collections::HashMap;
use std::iter;

use rand::Rng;

// For Types like `vec` which can have dimensions `vec2`, `vec3`, and `vec4`
#[derive(Debug, Clone)]
enum GLSLDimension {
    Two,
    Three,
    Four,
}

impl From<&GLSLDimension> for &str {
    fn from(dim: &GLSLDimension) -> Self {
        match dim {
            GLSLDimension::Two => "2",
            GLSLDimension::Three => "3",
            GLSLDimension::Four => "4",
        }
    }
}

// All possible GLSL types that are supported or that I should support
#[derive(Debug, Clone)]
enum GLSLType {
    // Non-vector types
    Bool,
    Int,
    Uint,
    Float,
    Double,
    // Vector-2 types
    BVec(GLSLDimension),
    IVec(GLSLDimension),
    UVec(GLSLDimension),
    Vec(GLSLDimension),
    DVec(GLSLDimension),
    // Matrix types
    Mat(GLSLDimension, GLSLDimension),
    DMat(GLSLDimension, GLSLDimension),
    // todo I think I need to describe arrays somehow?
    ArrayInt,
    ArrayUint,
    ArrayFloat,
    ArrayVec(GLSLDimension),
    // Sampler types todo add more https://www.khronos.org/opengl/wiki/Sampler_(GLSL)
    Sampler,
    // Shadow Sampler types todo add more https://www.khronos.org/opengl/wiki/Sampler_(GLSL)
    SamplerShadow,
    // Texture types  todo figure out all of the types that should be allowed here
    TextureCube,
    Texture2D,
    Texture2DArray,
}

impl Parse for GLSLType {
    fn parse(input: ParseStream) -> Result<Self> {
        let glsl_type = input.parse::<Ident>()?;
        let mut arr_num = 0;
        while !input.is_empty() {
            let _x;
            bracketed!(_x in input);
            arr_num += 1;
        }
        match (glsl_type.to_string().as_ref(), arr_num) {
            ("bool", 0) => Ok(GLSLType::Bool),
            ("int", 0) => Ok(GLSLType::Int),
            ("uint", 0) => Ok(GLSLType::Uint),
            ("float", 0) => Ok(GLSLType::Float),
            ("double", 0) => Ok(GLSLType::Double),
            ("vec2", 0) => Ok(GLSLType::Vec(GLSLDimension::Two)),
            ("vec3", 0) => Ok(GLSLType::Vec(GLSLDimension::Three)),
            ("vec4", 0) => Ok(GLSLType::Vec(GLSLDimension::Four)),
            ("mat4", 0) => Ok(GLSLType::Mat(GLSLDimension::Four, GLSLDimension::Four)),
            ("int", 1) => Ok(GLSLType::ArrayInt),
            ("uint", 1) => Ok(GLSLType::ArrayUint),
            ("float", 1) => Ok(GLSLType::ArrayFloat),
            ("vec2", 1) => Ok(GLSLType::ArrayVec(GLSLDimension::Two)),
            ("vec3", 1) => Ok(GLSLType::ArrayVec(GLSLDimension::Three)),
            ("vec4", 1) => Ok(GLSLType::ArrayVec(GLSLDimension::Four)),
            ("sampler", 0) => Ok(GLSLType::Sampler),
            ("samplerShadow", 0) => Ok(GLSLType::SamplerShadow),
            ("texture2D", 0) => Ok(GLSLType::TextureCube),
            ("texture2DArray", 0) => Ok(GLSLType::Texture2D),
            ("textureCube", 0) => Ok(GLSLType::Texture2DArray),
            (x, _) => {
                panic!("We currently do not support {}", x)
            }
        }
    }
}

// This is a parameter which is often written like :
// `[group1 [buffer loop in out] uint[]] indices`
// where `group1` is the name of the group for this parameter
// where `[buffer loop in out]` is a list of qualifiers
// where `uint[]` is the type
// where `indices` is the name of the parameter
#[derive(Debug, Clone)]
struct Parameters {
    group: Option<Ident>,
    quals: Vec<Ident>, //todo Add Qualifiers parsing instead of ident
    glsl_type: GLSLType,
    name: Ident,
}

impl PartialEq for Parameters {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for Parameters {}

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

        let glsl_type = qual_and_type.parse::<GLSLType>()?;

        let name = input.parse::<Ident>()?;
        Ok(Parameters {
            group,
            glsl_type,
            quals: quals.into_iter().collect(),
            name,
        })
    }
}

// Contains the parameters of a single glsl shader
// todo Later this should also contain the body of the shader
struct Shader {
    params: Vec<Parameters>, //body: String,
}

fn is_gl_builtin(p: &str) -> bool {
    let builtin = vec![
        "gl_VertexID",
        "gl_InstanceID",
        "gl_FragCoord",
        "gl_FrontFacing",
        "gl_PointCoord",
        "gl_SampleID",
        "gl_SamplePosition",
        "gl_NumWorkGroups",
        "gl_WorkGroupID",
        "gl_LocalInvocationID",
        "gl_GlobalInvocationID",
        "gl_LocalInvocationIndex",
    ];
    builtin.contains(&p)
}

impl Parse for Shader {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut params = Vec::new();
        while !input.peek(syn::token::Brace) {
            let p = input.parse::<Parameters>()?;
            if !is_gl_builtin(&p.name.to_string()) {
                params.push(p);
            }

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

// Contains the parameters of one or more shaders which make up a pipeline context
// Will be created for the user at `context`
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
            s.params.into_iter().for_each(|p| {
                if p.quals.contains(&format_ident!("in")) {
                    if !outs.contains(&p) && !ins.contains(&p) {
                        ins.push(p);
                    }
                } else if p.quals.contains(&format_ident!("out")) {
                    if ins.contains(&(p)) {
                        ins.remove(ins.iter().position(|x| *x == p).unwrap());
                    } else {
                        outs.push(p);
                    }
                }
            })
        });

        Ok(Context {
            context,
            ins: ins.into_iter().collect(),
            outs: outs.into_iter().collect(),
        })
    }
}

fn create_mat_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    // maybe other args with other mat types
) {
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
        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
            args: mat_type,
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    unit_type: syn::Ident,
) {
    data_type.push(syn::PathSegment {
        ident: unit_type,
        arguments: syn::PathArguments::None,
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
    n: &GLSLDimension,
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
                lit: syn::Lit::Int(syn::LitInt::new(n.into(), proc_macro2::Span::call_site())),
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
            colon2_token: None,
            lt_token: Token!(<)(proc_macro2::Span::call_site()),
            args: generic_args,
            gt_token: Token!(>)(proc_macro2::Span::call_site()),
        }),
    });
}

fn create_sampler_type(
    data_type: &mut syn::punctuated::Punctuated<syn::PathSegment, syn::token::Colon2>,
    qualifiers: &Vec<Ident>,
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
        ident: if qualifiers.contains(&format_ident!("compare")) {
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
        ident: if qualifiers.contains(&format_ident!("filter")) {
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
    view_dimension: syn::Ident,
    qualifiers: &Vec<Ident>,
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
    //todo push True if multisample
    multi_sample_path.push(syn::PathSegment {
        ident: if qualifiers.contains(&format_ident!("MultiSample")) {
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
    //todo push different dimension
    view_dimension_path.push(syn::PathSegment {
        ident: view_dimension,
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
    //todo push different type when needed
    sample_type_path.push(syn::PathSegment {
        ident: format_ident!("Float"),
        arguments: syn::PathArguments::None,
    });

    let mut sample_type_field = syn::punctuated::Punctuated::new();
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

fn create_base_type(ty: &GLSLType, qualifiers: &Vec<Ident>) -> syn::GenericArgument {
    let mut data_type = syn::punctuated::Punctuated::new();
    match ty {
        GLSLType::Float => {
            //Vec<f32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_type(&mut generic_type, format_ident!("f32"));
            create_buffer_type(
                &mut data_type,
                generic_type,
                qualifiers.contains(&format_ident!("buffer")),
            );
        }
        GLSLType::Vec(dim) | GLSLType::ArrayVec(dim) => {
            //Vec<[f32; dim]>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_vec_array_type(&mut generic_type, format_ident!("f32"), dim);
            create_buffer_type(
                &mut data_type,
                generic_type,
                qualifiers.contains(&format_ident!("buffer")),
            );
        }
        GLSLType::Mat(GLSLDimension::Four, GLSLDimension::Four) => {
            //cgmath::Matrix4<f32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_mat_type(&mut generic_type);
            create_buffer_type(
                &mut data_type,
                generic_type,
                qualifiers.contains(&format_ident!("buffer")),
            );
        }
        GLSLType::Int => {
            //Vec<i32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_vec_type(&mut generic_type, format_ident!("i32"));
            create_buffer_type(
                &mut data_type,
                generic_type,
                qualifiers.contains(&format_ident!("buffer")),
            );
        }
        GLSLType::Uint | GLSLType::ArrayUint => {
            //todo what is the difference between this data type and ArratUint
            // Vec<u32>
            let mut generic_type = syn::punctuated::Punctuated::new();
            create_vec_type(&mut generic_type, format_ident!("u32"));
            create_buffer_type(
                &mut data_type,
                generic_type,
                qualifiers.contains(&format_ident!("buffer")),
            );
        }
        GLSLType::Sampler => {
            create_sampler_type(&mut data_type, qualifiers);
        }
        //todo see what setting need to be different for this
        GLSLType::SamplerShadow => {
            create_sampler_type(&mut data_type, qualifiers);
        }
        GLSLType::Texture2D => {
            create_texture_type(&mut data_type, format_ident!("D2"), qualifiers);
        }
        GLSLType::Texture2DArray => {
            create_texture_type(&mut data_type, format_ident!("D2Array"), qualifiers);
        }
        GLSLType::TextureCube => {
            create_texture_type(&mut data_type, format_ident!("Cube"), qualifiers);
        }
        _ => panic!("Unsupported type: {:?}", ty),
    }

    syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: data_type,
        },
    }))
}

fn create_vertex(ty: &GLSLType, quals: &Vec<Ident>) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    bind_ty.push(create_base_type(ty, quals));

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

fn create_bindgroup(ty: Vec<(&GLSLType, &Vec<Ident>)>) -> syn::Type {
    let mut bind_ty = syn::punctuated::Punctuated::new();
    ty.iter()
        .for_each(|(t, q)| bind_ty.push(create_base_type(t, q)));

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

    // (name, type)
    let input_vec = shader_params.ins;

    let input_names: Vec<Ident> = input_vec.iter().map(|p| p.name.clone()).collect();
    let out_vec = shader_params.outs;
    let output_names: Vec<Ident> = out_vec.iter().map(|p| p.name.clone()).collect();

    let input_params = process_params(input_vec);

    let mut rng = rand::thread_rng();

    let n1: u8 = rng.gen();
    let context = format_ident!("Context{}", n1);

    let mut all_expanded = Vec::new();

    let variables: Vec<syn::Type> = (1..input_params.len() + 1)
        .into_iter()
        .map(|x| make_trait(format!("T{}", x)))
        .collect();
    let fields: Vec<Ident> = (1..input_params.len() + 1)
        .into_iter()
        .map(|x| format_ident!("field{}", x))
        .collect();
    let init: Vec<syn::Type> = iter::repeat(unbound()).take(input_params.len()).collect();
    let run: Vec<syn::Type> = iter::repeat(bound()).take(input_params.len()).collect();
    let ctxloc = shader_params.context;

    // For setting up pipeline
    let mut bind_group_types: Vec<(u32, syn::Type)> = input_params
        .clone()
        .into_iter()
        .filter_map(|a| match a {
            ParamType::Vertex { .. } => None,
            ParamType::Group { num, param } => Some((
                num,
                create_bindgroup(param.iter().map(|p| (&p.glsl_type, &p.quals)).collect()),
            )),
        })
        .collect();
    bind_group_types.sort_by(|(a, _), (b, _)| a.cmp(b));

    let sorted_bind_group_types: Vec<syn::Type> =
        bind_group_types.into_iter().map(|(_, x)| x).collect();

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
            fn get_layout(&self, device : &wgpu::Device) -> Vec<wgpu::BindGroupLayout> {
                vec![#(#sorted_bind_group_types::get_layout(device),)*]
            }
        }

        impl <'a,  T : pipeline :: RuntimePass<'a>> pipeline::ContextInputs for #context<'a, T, #(#init),*> {
            fn inputs(&self) -> Vec<String> {
                vec![#(stringify!(#input_names).to_string()),*]
            }
        }

        impl <'a,  T : pipeline :: RuntimePass<'a>> #context<'a, T, #(#run),*> {
            fn runnable<P, B>(&self, f: P) -> B where P: FnOnce() -> B{
                f()
            }
            fn can_pipe(&self, b : &dyn pipeline::ContextInputs) {
                let a = vec![#(stringify!(#output_names).to_string()),*];
                assert!(b.inputs().iter().all(|item| a.contains(item)));
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
                .fold("set".to_string(), |acc, p| format!("{}_{}", acc, p.name))
        );

        let index = index_numbers.get(current_thing.get_num() as usize).unwrap();

        let data_type = match current_thing.clone() {
            ParamType::Vertex { param, .. } => create_vertex(&param.glsl_type, &param.quals),
            ParamType::Group { param, .. } => {
                create_bindgroup(param.iter().map(|p| (&p.glsl_type, &p.quals)).collect())
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
        let restricted_abstract: Vec<syn::Type> = trait_params
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
        let restricted_trait: Vec<syn::Type> = trait_params
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
                trait #trait_name<'a,  T : pipeline :: RuntimePass<'a>, #(#trait_params: pipeline::AbstractBind,)* >{
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#type_params),*>;
                }

                impl<'a,  T : pipeline :: RuntimePass<'a>, #(#restricted_abstract: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#restricted_trait,)*> for &#context<'a, T, #(#restricted_impl),*> {
                    fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#restricted_type),*>{
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

            impl<'a,  T : pipeline :: RuntimePass<'a>, #(#restricted_abstract: pipeline::AbstractBind,)* > #trait_name<'a, T, #(#restricted_trait,)*> for &#context<'a, T, #(#restricted_impl),*> {
                fn #bind_name(self, rpass: &mut T, data : &'a #data_type) -> #context<'a, T, #(#restricted_type),*>{
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

    //println!("{}", collapsed_expanded);

    // Hand the output tokens back to the compiler
    TokenStream::from(collapsed_expanded)
}
