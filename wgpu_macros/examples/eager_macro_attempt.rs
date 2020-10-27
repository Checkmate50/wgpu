    macro_rules! find_params {
        ($context:tt {$($ins:tt,)*} {$($outs:tt,)*}) => {generic_bindings! {$context = $($ins),*; $($outs,)*}};
        ($context:tt [] $done_with:tt $([$($qualifier:tt)*] $param:tt)* {$($ins:tt,)*}{ $($outs:tt,)*}) => {find_params! {$context $([$($qualifier)*] $param)* {$($ins,)*}{ $($outs,)*}}};

        ($context:tt [in $($rest:tt)*] $current:tt $([$($qualifier:tt)*] $param:tt)* {$($ins:tt,)*}{ $($outs:tt,)*}) => {find_params! {$context [$($rest)*] $current $([$($qualifier)*] $param)* {$($ins,)* $current,}{ $($outs,)*}}};

        ($context:tt [out $($rest:tt)*] $current:tt $([$($qualifier:tt)*] $param:tt)* {$($ins:tt,)*}{ $($outs:tt,)*}) => {find_params! {$context [$($rest)*] $current $([$($qualifier)*] $param)* {$($ins,)*}{ $($outs,)* $current,}}};

        ($context:tt [$else:tt $($rest:tt)*] $current:tt $([$($qualifier:tt)*] $param:tt)* {$($ins:tt,)*}{ $($outs:tt,)*}) => {find_params! {$context [$($rest)*] $current $([$($qualifier)*] $param)* {$($ins,)*}{ $($outs,)*}}};
    }

    macro_rules! find_params_helper {
        ($context:tt {$([[$($qualifier:tt)*] $type:ident $($brack:tt)*] $param:ident;)*
            {$($tt:tt)*}}) => {find_params!{$context $([$($qualifier)*] $param)* {}{}};}
    }

    macro_rules! my_shader {
        ($name:tt = $($tt:tt)*) => {
            macro_rules! $name{
                (shader) => {compute_shader! $($tt)* };
                ($context:tt) => {find_params_helper!{$context $($tt)*}};
            }
        };
    }

    my_shader!(compute = {
        [[buffer loop in out] uint[]] indices;
        [[buffer in] uint[]] indices2;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                indices[index] = indices[index]+indices2[index];
            }
        }}});

    /* const S: ComputeShader = compute!(shader);

    compute!(context); */