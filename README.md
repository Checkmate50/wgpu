# wgpu

## Binding

### Introduction

Binding is the assigning of data to a buffer on the Gpu. The Cpu representations of these buffers(+parameter meta data) are stored in a set of bindings. When a program(a pipeline compiled from a compute shader or vertex/fragment shader pair) is ready to be run, the buffers are assigned to bindings of the pipeline and the pipeline is drawn/dispatched.

### Validity conditions

The binding of data to a parameter in a program is valid if:

- The name that is being bound to is the name of a parameter in the program.
- The parameter to be bound to is specified by an ```in``` qualifier.
- The type of the data being bound is equivalent to the type of the parameter.
- The parameter has not already been bound.

An execution of a program within a binding context is valid if:

- All parameters with an ```in``` qualifier have been bound to.
- The data bound to a parameter has not been modified.

### Managing the binding conditions

At the moment, the following structures enforce these conditions: a binding context, and two sets of bindings.

todo update binding context to talk about param_types
The ([binding context](wgpu_macros/lib.rs)) is a series of macro generated structs roughly of the form ```Context<AbstractBind, ...>``` 

todo remove old binding_context description
The ([binding context](src/context.rs)) maintains a list of ```in``` parameters that need to be bound before the program is run and a list of ```out``` parameters that will be produced as a result of the program. When a parameter is bound, its name is removed from the list of ```in``` parameters. When the list is empty, the program is ready to be run. When a parameter is bound that is also a member of the ```out``` parameter list, the usability(as tracked by the usage context) has been affected and a flag is flipped.

Finally, there are two sets of ([bindings](src/shared.rs)) which collect the buffers as they are generated. Currently there are two, ```ProgramBindings``` to collect ```in``` qualified parameters and ```OutProgramBindings``` to collect ```out``` parameters. They also handle the metadata of the parameter and check that the data being bound is of the right type.

### Typing
todo update this to talk about param types
The usage context has two types(```Context``` and ```MutContext```) to linearly track the usability of the context. This means we need three functions to bind: ```bind: Context -> Context```, ```bind_mutate: Context -> MutContext```, and ```bind_consume: MutContext -> MutContext```. The interesting function here is ```bind_mutate``` since that function converts the multi-use ```Context``` into a single-use ```MutContext```. This must be used the first time a parameter annotated with ``in out`` qualifiers is bound since the data within the context will change after the program has been run.

Most data is bound as an array of bytes. Common types of data have implementations of the ```Bindable``` trait which handles the context type and conversion of data to ```&[u8]```. Binding samplers and textures require different bind functions.

## The docs

<https://docs.rs/wgpu/0.5.0/wgpu>

The underlying target of the rust bindings which is sometimes useful if the wgpu documentation is not sufficient.
<https://gpuweb.github.io/gpuweb/>

For a comparison with metal
<https://developer.apple.com/documentation/metal/>

Windows/display related stuff
<https://docs.rs/winit/0.22.2/winit>

## Resources

A decent tutorial that breaks up it's example into bite-sized pieces but adds very little that can't be learned from reading the documentation.
<https://sotrh.github.io/learn-wgpu/>

A more explanatory tutorial
<https://alain.xyz/blog/raw-webgpu>

A list of other tutorials/examples/use cases of wgpu
<https://github.com/rofrol/awesome-wgpu>