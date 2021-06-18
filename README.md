# wgpu

## Binding

### Introduction

Binding is the assigning of data to a buffer on the GPU. The CPU representations of these buffers(+parameter metadata) are stored as special buffer structs or in bind groups of buffers (see [bind.rs](src/bind.rs)). During a given iteration, these can then be set in the context of a pipeline. When a program(a pipeline compiled from a compute shader or vertex/fragment shader pair) has a completely bound context, the pipeline is drawn/dispatched.

### Validity conditions

The binding of data to a parameter in a program is valid if:

- The name that is being bound to is the name of a parameter in the program.
- The parameter to be bound to is specified by an ```in``` qualifier.
- The type of the data being bound is equivalent to the type of the parameter.
- The parameter has not already been bound.
- If the parameter is part of group, the group of data is bound together with the same layout/types as the group of parameters

An execution of a program within a binding context is valid if:

- All parameters with an ```in``` qualifier have been bound to.
- The data bound to a parameter has not been modified.

### Managing the binding conditions

At the moment, the following structures enforce these conditions: a binding context, an Index buffer struct, a Vertex buffer struct, and BindGroup structs of various dimensions.

The binding context is created for the programmer by macros for their specific set of shaders. If comes in the form Context<A, B, ...> where it is parameterized across the expected inputs for the shader and whether they are bound or not. It also maintains a list of its outputs to be able to check whether it can be used to pipe into the inputs of the next program. For each parameter, a trait is implemented for this Context that turns it from the Unbound to Bound state.

The data passed to this trait is of two forms, a Vertex struct or a BindGroup struct depending on the data type of the shader. Each of these is parameterized over the types of its inputs. Only a Vertex/BindGroup struct of the correct type will be accepted.

Index data is optional data stored in a buffer on the GPU to access the correct index in the Vertex Struct for each iteration of the vertex shader.

## An example

Take for example this vertex fragment pair.

```rust
my_shader! {vertex = {
    [[vertex in] vec3] a_position;
    [[vertex in] vec3] vertexColor;

    [group1 [uniform in] mat4] u_view;
    [group1 [uniform in] mat4] u_proj;

    [[out] vec3] fragmentColor;
    [[out] vec4] gl_Position;

    {{
       ...
    }}
}}

my_shader! {fragment = {
    [[in] vec3] fragmentColor;
    [[out] vec4] color;
    {{
        ...
    }}
}}
```

The shorthand for how to set up this pipeline without this library:

```rust
let pass = render_pass_initializer(...);

pass.set_vertex_buffer(0, color_data); // color_data : vec3
pass.set_vertex_buffer(1, position_data); // position_data : vec4
pass.set_bind_group(0, u_view_proj_mat); // u_view_proj_mat : mat4

pass.run();
```

This has allowed for a bunch of bugs! Here is a similar shorthand for how to set up the same pipeline in this library:

```rust
let context = context_initializer(...);

// Each Vertex struct is parameterized on its input data to carry it's type
let position_vertex = Vertex(position_data); // must be vec3 to typecheck
let color_vertex = Vertex(color_data);
// Bindgroups are typed based on the number of inputs
let view_proj_bind_group = BindGroup2(u_view_mat, u_proj_mat);

...
let pass = render_pass_initializer(...);
{
    // When you set each vertex struct, its typechecked to make sure the input data is of the expected type for the shader.
    context2 = context.set_a_position(pass, position_vertex);
    {
        // Since each function is created for that parameter, it already knows which slot
        // in the pipeline the data goes.
        context3 = context2.set_vertexColor(pass, color_vertex);
        {
            // Only BindGroup2's with the right parameterized types are accepted
            context4 = context3.set_u_view_u_proj(pass, view_proj_bind_group);
            {
                // Here we can statically check that all of the inputs for the pass have been bound
                context4.run(pass);
            }
        }
    }
}
```

## Future Work

### Immediate Work

- User-defined structs.
- We can probably move more to proc macros to get rid of brackets
- Better/more testing.
- Using real projects.
- Create documentation.
- Clean the project up and make it performant.

### Distant Work

- Enforce scope around Context<>'s without relying on the programmer.
- Testing the modularity of compute shaders. See if there are examples where this library can make it easier to create pipelines of shaders.
- Compute shaders should be able to dispatch a job across more than just a single dimension. Currently, we go one by one in the x-direction.
- Look into how more optimizations can be applied. For instance, are there optimizations around how multiple pieces of data can be stored in the same buffer?
- Interloping with Gator?

## Limitations
- The current libraries are fragile.
<!--- todo what does fragile mean??? --->
- Data cannot be shared across bind groups.
- The programmer is responsible for choosing appropriate/efficient bind groups.
- The programmer is responsible for maintaining that a given render pass is only used with its corresponding context.
- Binding to Context's in a tree-like scope will be worse than manually binding to the render pass in some contrived cases.
- Checking for mutation completely relies on the programmer annotating a parameter as ```[in out]```. We currently can't check when this should be the case but isn't.
- Since bind groups get created separately, you can't optimize the program to use one uniform buffer for multiple groups and use indexing to switch between them.
- There are only a couple sets of defaults a library exposes to the user. This means programmers are limited significantly in how they can tweak the render pass settings. For instance, you are not currently able to take a multi-sampling approach.

## The docs

<https://docs.rs/wgpu/0.5.0/wgpu>

The underlying target of the rust bindings which is sometimes useful if the wgpu documentation is not sufficient.
<https://gpuweb.github.io/gpuweb/>

For a comparison with metal
<https://developer.apple.com/documentation/metal/>

Windows/display related stuff
<https://docs.rs/winit/0.22.2/winit>

## Resources

A decent tutorial that breaks up its example into bite-sized pieces but adds very little that can't be learned from reading the documentation. It does have useful information on transitioning across breaking version changes and is continuously being improved.
<https://sotrh.github.io/learn-wgpu/>

A more explanatory tutorial
<https://alain.xyz/blog/raw-webgpu>

A list of other tutorials/examples/use cases of wgpu
<https://github.com/rofrol/awesome-wgpu>