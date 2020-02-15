#version 450

layout(location = 0) in vec3 frag_pos;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(set = 0, binding = 3) uniform Ambient_Data {
    vec3 color;
    float intensity;
} ambient;

layout(set = 0, binding = 4) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 ambient_color = ambient.intensity * ambient.color;
    vec3 light_direction = normalize(directional.position.xyz - frag_pos);
    float directional_intensity = max(dot(normalize(subpassLoad(u_normals).rgb), light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = (ambient_color + directional_color) * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
}
