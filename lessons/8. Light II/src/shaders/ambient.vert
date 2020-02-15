#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 2) uniform MVP_Data {
    mat4 model;
    mat4 view;
    mat4 projection;
} uniforms;

layout(location = 0) out vec3 frag_pos;

void main() {
    mat4 worldview = uniforms.view * uniforms.model;
    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
    frag_pos = vec3(uniforms.model * vec4(position, 1.0));
}