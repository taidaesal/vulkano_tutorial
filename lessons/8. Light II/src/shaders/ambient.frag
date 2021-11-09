#version 450

layout(location = 0) in vec3 frag_pos;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(set = 0, binding = 3) uniform Ambient_Data {
    vec3 color;
    float intensity;
} ambient;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 combined_color = ambient_color * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
}
