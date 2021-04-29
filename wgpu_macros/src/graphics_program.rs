use proc_macro::TokenStream;
use proc_macro2::Ident;
use syn::{parse_macro_input, LitStr};
use quote::{quote, format_ident};

use naga::{valid::{ValidationFlags, Validator}, front::wgsl, GlobalVariable, Module, ShaderStage, Binding};

#[derive(Debug, Clone)]
enum Input {
    Argument { binding: u32, name: String },
    Uniform { binding: u32, name: String },
}

impl Input {
    fn binding(&self) -> u32 {
        match self {
            Input::Argument { binding, .. } | Input::Uniform { binding, .. } => *binding,
        }
    }

    fn name(&self) -> &str {
        match self {
            Input::Argument { name, .. } | Input::Uniform { name, .. } => name,
        }
    }
}

fn make_inputs(module: &Module) -> Vec<Option<Input>> {
    let vert_inputs = module.entry_points.iter()
        .find(|x| x.stage == ShaderStage::Vertex).unwrap().function.arguments.iter()
        .filter_map(|arg| Some(arg.binding.as_ref()
                    .map(|binding| {
                        match binding {
                            Binding::Location { location, .. } => Some(Input::Argument { binding: *location, name: arg.name.clone().unwrap() }),
                            Binding::BuiltIn(..) => None,
                        }
        }).flatten()));
    let vars = module.global_variables.iter().map(|v| v.1.clone()).collect::<Vec<GlobalVariable>>();
    let mut grouped_vars = vec![Vec::new(); 1 + (&vars).last().unwrap().binding.as_ref().unwrap().group as usize];
    for var in vars {
        let binding = var.binding.as_ref().unwrap().group;
        let input = Input::Uniform { binding, name: var.name.unwrap() };
        grouped_vars[binding as usize].push(input);
    }
    let uniforms = grouped_vars.into_iter().map(|x| x.into_iter().reduce(|a, b| {
        Input::Uniform { binding: a.binding(), name: format!("{}_{}", a.name(), b.name()) }
    }));
    uniforms.chain(vert_inputs).collect::<Vec<Option<Input>>>()
}

pub fn sub_module_graphics_program(input: TokenStream) -> TokenStream {
    let shader_str = parse_macro_input!(input as LitStr).value();

    // TODO: make these errors better
    // this is the entirety of shader parsing and validation
    let shader_module = wgsl::parse_str(&shader_str).expect("Failed to parse shader");
    Validator::new(ValidationFlags::all()).validate(&shader_module).expect("Invalid shader");

    let inputs = make_inputs(&shader_module);

    let final_context_name = format_ident!("Context{}", "1".repeat((&inputs).len()));
    let initial_context_name = format_ident!("Context{}", "0".repeat((&inputs).len()));
    let impls = make_context(inputs);

    let pipeline = make_pipeline(&shader_str, shader_module);

    let contexts = quote! {
        #impls

        struct #final_context_name<'a> {
            rpass: &'a mut wgpu::RenderPass<'a>,
        }
        impl #final_context_name<'_> {
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

            pub fn start_pass(&'a mut self, encoder: &'a wgpu::CommandEncoder, desc: wgpu::RenderPassDescriptor<'a, '_>) -> #initial_context_name<'a> {
                let mut rpass = encoder.begin_render_pass(&desc);
                rpass.set_pipeline(&self.pipeline);
                self.rpass = Some(rpass);

                #initial_context_name {
                    rpass: &mut rpass,
                }
            }
        }

        let program = GraphicsProgram { pipeline, rpass: None };
    };

    TokenStream::from(contexts)
}

fn make_context(inputs: Vec<Option<Input>>) -> proc_macro2::TokenStream {
    let mut impls = quote!();
    let mut contexts = quote!();

    for (i, maybe_input) in inputs.iter().enumerate() {
        if let Some(input) = maybe_input {
            let mut new_vars = inputs.clone();
            new_vars[i] = None;
            let new_name = context_name_from_vars(&new_vars);
            let lower_context = make_context(new_vars);
            contexts = quote! {
                #contexts
                #lower_context
            };
            match input {
                Input::Argument { binding, name } => {
                    let setter_name = format_ident!("set_{}", name);
                    impls = quote! {
                        #impls

                        fn #setter_name(&'a mut self, bind_group: &'a wgpu::BindGroup) -> #new_name {
                            self.rpass.set_bind_group(#binding, bind_group, &[]);
                            #new_name { rpass: self.rpass }
                        }
                    }
                },
                Input::Uniform { binding, name } => {
                    let setter_name = format_ident!("set_{}", name);
                    impls = quote! {
                        #impls

                        fn #setter_name(&'a mut self, buffer_slice: wgpu::BufferSlice<'a>) -> #new_name {
                            self.rpass.set_vertex_buffer(#binding, buffer_slice);
                            #new_name { rpass: self.rpass }
                        }
                    }
                }
            }
        }
    }

    if !impls.is_empty() {
        let context_name = context_name_from_vars(&inputs);
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

fn context_name_from_vars(vars: &Vec<Option<Input>>) -> Ident {
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
                module: &module,
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
