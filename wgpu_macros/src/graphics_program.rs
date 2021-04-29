use proc_macro::TokenStream;
use proc_macro2::Ident;
use syn::{parse_macro_input, LitStr};
use quote::{quote, format_ident};

use naga::{valid::{ValidationFlags, Validator}, front::wgsl, GlobalVariable, Module, ShaderStage};

pub fn sub_module_graphics_program(input: TokenStream) -> TokenStream {
    let shader_str = parse_macro_input!(input as LitStr).value();

    // TODO: make these errors better
    // this is the entirety of shader parsing and validation
    let shader_module = wgsl::parse_str(&shader_str).expect("Failed to parse shader");
    Validator::new(ValidationFlags::all()).validate(&shader_module).expect("Invalid shader");

    let mut vars: Vec<Option<GlobalVariable>> = shader_module.global_variables.iter().map(|v| Some(v.1.clone())).collect();
    vars.dedup_by_key(|v| v.as_ref().unwrap().binding.as_ref().unwrap().group);

    let impls = make_context_with_vars(vars);

    let pipeline = make_pipeline(&shader_str, shader_module);

    let contexts = quote! {
        #impls

        struct Context11<'a> {
            rpass: &'a mut wgpu::RenderPass<'a>,
        }
        impl Context11<'_> {
            pub fn draw(&mut self, vertices: core::ops::Range<u32>, instances: core::ops::Range<u32>) {
                self.rpass.draw(vertices, instances);
            }
        }

        #pipeline

        struct GraphicsProgram<'a> {
            pipeline: wgpu::RenderPipeline,
            rpass: Option<wgpu::RenderPass<'a>>,
        }

        impl<'a> GraphicsProgram<'a> {

            pub fn start_pass(&'a mut self, encoder: &'a wgpu::CommandEncoder, desc: wgpu::RenderPassDescriptor<'a, '_>) -> Context00<'a> {
                let mut rpass = encoder.begin_render_pass(&desc);
                rpass.set_pipeline(&self.pipeline);
                self.rpass = Some(rpass);

                Context00 {
                    rpass: &mut rpass,
                }
            }
        }

        let program = GraphicsProgram { pipeline, rpass: None };
    };

    TokenStream::from(contexts)
}

fn make_context_with_vars(vars: Vec<Option<GlobalVariable>>) -> proc_macro2::TokenStream {
    let mut impls = quote!();
    let mut contexts = quote!();

    for (i, maybe_var) in vars.iter().enumerate() {
        if let Some(var) = maybe_var {
            let setter_name = format_ident!("set_{}", var.name.as_ref().unwrap());
            let group_num = var.binding.as_ref().unwrap().group;
            let mut new_vars = vars.clone();
            new_vars[i] = None;
            let new_name = context_name_from_vars(&new_vars);
            let lower_context = make_context_with_vars(new_vars);
            contexts = quote! {
                #contexts
                #lower_context
            };
            impls = quote! {
                #impls

                fn #setter_name(&'a mut self, bind_group: &'a wgpu::BindGroup) -> #new_name {
                    self.rpass.set_bind_group(#group_num, bind_group, &[]);
                    #new_name { rpass: self.rpass }
                }
            }
        }
    }

    if !impls.is_empty() {
        let context_name = context_name_from_vars(&vars);
        let struct_name = quote!{
            struct #context_name<'a> {
                rpass: &'a mut wgpu::RenderPass<'a>,
            }
        };

        let final_tokens = quote! {
            #contexts

            #struct_name

            impl<'a> #context_name<'a> {
                #impls
            }
        };

        final_tokens
    } else {
        impls
    }
}

fn context_name_from_vars(vars: &Vec<Option<GlobalVariable>>) -> Ident {
    let nums = vars.iter().map(|x| match x {
        Some(_) => '0',
        None => '1',
    }).collect::<String>();

    format_ident!("Context{}", nums)
}

fn make_pipeline(shader_str: &str, shader_module: Module) -> proc_macro2::TokenStream {
    let vert_entry_point = shader_module.entry_points.iter().find(|x| x.stage == ShaderStage::Vertex).map(|x| x.name.clone()).unwrap();
    let frag_entry_point = shader_module.entry_points.into_iter().find(|x| x.stage == ShaderStage::Fragment).map(|x| x.name);
    let frag_module = if let Some(entry_point) = frag_entry_point {
        quote! {
            let fragment = Some(wgpu::FragmentState {
                module,
                entry_point: #entry_point,
                targets: &[],
            });
        }
    } else {
        quote! {
            let fragment = None;
        }
    };
    let result = quote! {
        let module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(#shader_str)),
            flags: wgpu::ShaderFlags::VALIDATION,
            label: None,
        });

        #frag_module

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: #vert_entry_point,
                buffers: &[],
            },
            fragment,
            primitive: wgpu::PrimitiveState::default(),
            multisample: wgpu::MultisampleState::default(),
            depth_stencil: None,
        });
    };

    result
}
