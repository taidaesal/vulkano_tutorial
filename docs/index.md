# Vulkano Tutorial

This site is intended to serve as a tutorial for using Vulkan in Rust via the Vulkano crate. This mostly documents my own experiments with learning graphical programming but I’m writing it down here in case others might find it useful. The Vulkano crate is in active development and so regularly experiences breaking changes. I will try to keep the examples updated but this is a hobby project.

Note: All code is provided under the [MIT License](http://opensource.org/licenses/MIT) and includes samples taken from official Vulkano examples. All text and images are licensed under the [Creative Commons Attribution](https://creativecommons.org/licenses/by/4.0/) license (CC BY 4.0)

## Lessons

### 0. Introduction
Provides a brief overview of some Vulkan considerations as well as a couple of notes on Rust.

[Introduction to Vulkan](./section_0.md)

### 1. Our first window

For our initial project, we will open a black window. This terminally-boring example is actually much longer and more important than you might expect. It will introduce most of the critical aspects shared by any Vulkan program.

[First Window](./section_1.md)

### 2. Rendering a triangle

With our first project we learned how to set up and use a Vulkan program. Now in this lesson we learn how to render things to the screen. This will involve writing our first shaders as well as passing information to them.

[Triangle](./section_2.md)

### 3. Transformations

After getting your first triangle on the screen the obvious question becomes "great, so how do I make it do things?" For that we will need to learn to apply *transformations* via `Uniform` data, which will be the other main way we feed data to our shaders in addition to the vertex data we learned about in the last lesson.

[Transformations](./section_3.md)

### 4. Depth

A shorter lesson that shows how to enable depth-testing in Vulkan. This is a necessity for the more complicated scenes we'll be creating in later lessons.

[Depth](./section_4.md)

### 5. Winding order

This lesson is a short one but it talks about a very important subject for rendering more complicated scenes: winding order and how you can use it to lower the amount of work your graphics hardware has to do.

[Winding Order](./section_5.md)

### 6. Light I

And in the sixth lesson the programmer said "let there be light" and lo, there was light!

[Light](./section_6.md)

### 7. Multi-pass rendering

Now that we're starting to get into more advanced features we're going to need to increase the power of our rendering system. We will accomplish this by introducing multiple rendering passes into the equation.

[Multi-pass rendering](./section_7.md)

### 8. Light II

Now that we've got our deferred rendering system set up, let's look at how you can use it to add more than one directional light to our scene.

[Multiple Lights](./section_8.md)

### 9. Models

Now that we've set up a working lighting system let's take a step back and look at something else that would be important to any graphics application: how to load models from a file.

[Loading Models](./section_9.md)

### 10. New Uniforms

After a couple of longer lessons, let's take a short break to refactor our normals code as well as set the stage for multiple models.

[Normal Uniform](./section_10.md)

### 11. Render System I

We're ready now to take our first stab at making a real rendering system. We won't introduce any new concepts in this lesson, just wrap up what we already know in a way that lets us input models to render as well as directional light sources. Think of this as a draft version of the more advanced system we'll make in a few lessons.

[Render System](./section_11.md)

#### 11.5 Light Objects

Let's take a bit of a break from "big picture" topics and implement a small feature that will give us the option to visually display where our light sources are in our scene.

[Light Objects](./section_11_5.md)

### 12. Light III

We're almost ready to move on to "advanced" topics, but first we have to revisit light one more time to complete our lighting model. This time we'll be adding "shininess."

[Specular Reflection](./section_12.md)

### 13. Textures

A brief look at textures and a discussion of their uses. 

[Textures](./section_13.md)

### 14. Text

A look at how to use textures to display a timer to the user.

[Text](./section_14.md)