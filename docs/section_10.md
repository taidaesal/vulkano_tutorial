---
layout: default
title: Uniform Refactoring
back_link: "./section_9.html"
next_link: "./section_11.html"
---

# Uniform Refactoring

Up until now, we've been using our regular MVP matrix to transform all our geometry data as well as our normals. However, this poses two problems:
1. Non-uniform transformations can create distorted normal vertices which will create problems with applying lighting to the scene.
2. The same transformations are being applied to all vertices, which means we can't have more than one model at a time.

The solution to both of these issues is to create a second uniform for values unique to each individual model in our scene. Inside this second uniform we will store the model matrix from our current MVP uniform, as well as a new matrix to represent the normal transformation matrix.

#### Changing MVP

We're moving the model matrix out our MVP uniform so let's update our MVP struct.
```rust
#[derive(Debug, Clone)]
struct VP {
    view: TMat4<f32>,
    projection: TMat4<f32>
}

impl VP {
    fn new() -> VP {
        VP {
            view: identity(),
            projection: identity()
        }
    }
}
```

#### Normal Matrix

The way to get the proper normal transformation matrix from the regular model matrix is to take the inverse of the model matrix and then transpose it. We can do this just fine inside our shaders but those operations are expensive so let's move it to the `Model` struct instead.

This is actually an interesting problem that will come up many times while doing graphics programming. Most non-trivial applications have more calculations that they'd *like* to do than they have the resources to actually do. So getting good performance is, in large part, deciding where to pay the cost of doing calculations you have to do. In many cases, math can be offloaded to our shaders since our graphics hardware is designed specifically do to the sorts of linear algebra operations we're discussing in this sub-section. However, we need to stop to consider that any calculation we do in our vertex shader will run on every single vertex we give it, the count of which could easily reach the tens or hundreds of thousands in non-trivial scene. So in this case, even though our shader could find the inverse transpose of our model matrix a thousand times faster than our CPU-bound application (at the very least) it's actually usually better to do it once on the CPU and pass it in to the shaders as a uniform.

`model.rs`
```rust
pub struct Model {
    data: Vec<NormalVertex>,
    translation: TMat4<f32>,
    rotation: TMat4<f32>,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    requires_update: bool,
}

pub fn model_matrices(&mut self) -> (TMat4<f32>, TMat4<f32>) {
    if self.requires_update {
        self.model = self.translation * self.rotation;
        self.normals = inverse_transpose(self.model);
        self.requires_update = false;
    }
    (self.model, self.normals)
}
```

`main.rs`
```rust
let (model_mat, normal_mat) = model.model_matrices();
```

#### Change our VP Buffer

So far all our uniforms have been created using `CpuBufferPool`. This is because all our uniforms have been updated every frame so we want something that supports frequent updates. However, now that we've turned our MVP structure into a VP one, we have a situation where one of our uniform inputs *won't* need to be updated. Our view and projection matrices will only need to be updated when the screen is resized.

Right now we re-declare `deferred_set` every frame. This was fine when the data it held was also changing every frame but now that we've changed that we need to stop recreating the associated descriptor set every frame as well.

Let's move our initial declaration to just above our main rendering loop. While we're at it, let's help keep the code readable by re-naming `deferred_set` to something more descriptive, like `vp_set`
```rust
let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
let mut vp_set = PersistentDescriptorSet::new(
    deferred_layout.clone(),
    [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
)
.unwrap();
```

We can re-create it right after we recreate our `vp_buffer` in the swapchain recreation portion of the render loop.
```rust
if recreate_swapchain {
    // ...

    let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
    vp_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        vp_layout.clone(),
        [WriteDescriptorSet::buffer(0, new_vp_buffer)],
    )
    .unwrap();

    recreate_swapchain = false;
}
```

With this, we maximize the performance gains of not needing to update our VP data each frame.

Replace the existing mvp_buffer declaration with the following code. Also remember to add it into our swapchain recreation logic.
```rust
let vp_buffer = CpuAccessibleBuffer::from_data(
    &memory_allocator,
    BufferUsage {
        uniform_buffer: true,
        ..BufferUsage::empty()
    },
    false,
    deferred_vert::ty::VP_Data {
        view: vp.view.into(),
        projection: vp.projection.into(),
    },
)
.unwrap();
```

```rust
if recreate_swapchain {
    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
    let image_extent: [u32; 2] = window.inner_size().into();

    let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
    vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
        image_extent,
        ..swapchain.create_info()
    }) {
        Ok(r) => r,
        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
    };

    let (new_framebuffers, new_color_buffer, new_normal_buffer) =
        window_size_dependent_setup(
            &memory_allocator,
            &new_images,
            render_pass.clone(),
            &mut viewport,
        );

    swapchain = new_swapchain;
    framebuffers = new_framebuffers;
    color_buffer = new_color_buffer;
    normal_buffer = new_normal_buffer;

    let new_vp_buffer = CpuAccessibleBuffer::from_data(
        &memory_allocator,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        deferred_vert::ty::VP_Data {
            view: vp.view.into(),
            projection: vp.projection.into(),
        },
    )
    .unwrap();

    let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
    vp_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        vp_layout.clone(),
        [WriteDescriptorSet::buffer(0, new_vp_buffer)],
    )
    .unwrap();

    recreate_swapchain = false;
}
```

To be perfectly honest, this is a small saving. But it's the sort of thing we need to keep in mind if we want to squeeze every bit of bandwidth possible out of our graphics hardware.

#### Updated Shaders

We just need to make changes to one shader. The changes themselves are small but significant.

`deferred.vert`
```glsl
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 view;
    mat4 projection;
} vp_uniforms;

layout(set = 1, binding = 0) uniform Model_Data {
    mat4 model;
    mat4 normals;
} model;

void main() {
    gl_Position = vp_uniforms.projection * vp_uniforms.view * model.model * vec4(position, 1.0);
    out_color = color;
    out_normal = mat3(model.normals) * normal;
}
```

On first glance this seems like the other times we've used multiple uniforms in our shaders but take another look at `Model_Data`.

This is the first time we've seen more than one set of uniforms in the same shader. This is because the set containing the VP data only needs to be updated when our VP data is changed but the set containing our model data will need to be refreshed every frame. This will require some changes to our main application though, so let's leave our shaders and go back to Rust-land.

#### Our new Model Data Descriptor Set

Creating a descriptor set for our model data is pretty simple. First we need to create a new `CpuBufferPool` to hold the uniform type.
```rust
let model_uniform_buffer: CpuBufferPool<deferred_vert::ty::Model_Data> =
    CpuBufferPool::uniform_buffer(memory_allocator.clone());
```

The following code can go where we used to declare our combined uniform sub-buffer and descriptor set.
```rust
let model_subbuffer = {
    let elapsed = rotation_start.elapsed().as_secs() as f64
        + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
    let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;
    model.zero_rotation();
    model.rotate(pi(), vec3(0.0, 1.0, 0.0));
    model.rotate(elapsed_as_radians as f32 * 50.0, vec3(0.0, 0.0, 1.0));
    model.rotate(elapsed_as_radians as f32 * 30.0, vec3(0.0, 1.0, 0.0));
    model.rotate(elapsed_as_radians as f32 * 20.0, vec3(1.0, 0.0, 0.0));

    let (model_mat, normal_mat) = model.model_matrices();

    let uniform_data = deferred_vert::ty::Model_Data {
        model: model_mat.into(),
        normals: normal_mat.into(),
    };

    model_uniform_buffer.from_data(uniform_data).unwrap()
};
```

```rust
let model_layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();
let model_set = PersistentDescriptorSet::new(
    &descriptor_set_allocator,
    model_layout.clone(),
    [WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
)
.unwrap();
```

The only real thing to notice is that we used `1` as the argument to `set_layouts.get()` for the first time in this lesson series.

#### Updating our Draw Command

Our draw command for the first render pass is pretty much the same as before. However, now we need to pass it two descriptor sets instead of one. The way to do this is fairly straightforward.

```rust
.bind_pipeline_graphics(deferred_pipeline.clone())
.bind_descriptor_sets(
    PipelineBindPoint::Graphics,
    deferred_pipeline.layout().clone(),
    0,
    (vp_set.clone(), model_set.clone()),
)
.bind_vertex_buffers(0, vertex_buffer.clone())
.draw(vertex_buffer.len() as u32, 1, 0, 0)
.unwrap()
```

As you can see, we just pass a list of the descriptor sets as a tuple, inside of parentheses.

#### Run the Code

Let's run our application. Nothing should have changed in the rendered output but it's always good to check before moving on.

![a picture of the Utah Teapot using the same lighting scheme as the previous lesson](./imgs/10/teapot.png)

I've swapped out Suzanne for the famous Utah Teapot but, as you can see, everything seems to be working as it should.

[lesson source code](https://github.com/taidaesal/vulkano_tutorial/tree/gh-pages/lessons/10.%20Uniform%20Refactoring)
