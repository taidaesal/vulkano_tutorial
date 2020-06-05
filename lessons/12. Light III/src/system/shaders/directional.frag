#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 1, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 1, set = 0, binding = 3) uniform subpassInput u_specular;

layout(set = 0, binding = 4) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 normal = subpassLoad(u_normals).xyz;
    float specular_intensity = subpassLoad(u_specular).x;
    float specular_shininess = subpassLoad(u_specular).y;
    vec3 view_dir = -normalize(camera.position - subpassLoad(u_frag_location).xyz);
    vec3 light_direction = normalize(directional.position.xyz + normal);
    vec3 reflect_dir = reflect(-light_direction, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), specular_shininess);
    vec3 specular = specular_intensity * spec * directional.color;
    float directional_intensity = max(dot(normal, light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;
    vec3 combined_color = (specular + directional_color) * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
}
