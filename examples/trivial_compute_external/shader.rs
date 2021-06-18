my_shader! {compute = {
        [group1 [buffer loop in] uint[]] indices;
        [group2 [buffer in out] uint[]] indices2;
        //[[buffer out] uint[]] result;
        //[... uint] xindex;
        {{
            void main() {
                // uint xindex = gl_GlobalInvocationID.x;
                uint index = gl_GlobalInvocationID.x;
                indices2[index] = indices[index]+indices2[index];
            }
        }}
    }}