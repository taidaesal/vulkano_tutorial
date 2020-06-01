#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(set = 0, binding = 2) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 dir = directional.position.xyz;
    vec3 norm = -subpassLoad(u_normals).xyz;
    vec3 light_direction = normalize(dir + subpassLoad(u_normals).xyz);
    //light_direction.y = -light_direction.y;
    float directional_intensity = max(dot(subpassLoad(u_normals).rgb, light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = directional_color * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
}
