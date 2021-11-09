#version 450
layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_location;

layout(set = 1, binding = 1) uniform Specular_Data {
    float intensity;
    float shininess;
} specular;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_location;
layout(location = 3) out vec2 f_specular;

void main() {
    f_color = vec4(in_color, 1.0);
    f_normal = in_normal;
    f_location = in_location;
    f_specular = vec2(specular.intensity, specular.shininess);
}