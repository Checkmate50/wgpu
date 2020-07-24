#version 450

layout(location=0) in vec3 posColor;
layout(location=1) in float brightness;
layout(location=0) out vec4 color;

void main() {
    color = vec4(posColor * brightness, 1.0);
}