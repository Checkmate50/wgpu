#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in float in_brightness;
layout(location=0) out vec3 posColor;
layout(location=1) out float brightness;

void main() {
    posColor = a_position;
    brightness = in_brightness;
    gl_Position = vec4(a_position, 1.0);
}