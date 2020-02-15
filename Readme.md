# Vulkano Tutorial

A repo for a tutorial series on Vulkan programming using the Vulkano Rust crate. The lessons go from initial project setup to toy rendering systems.

## Introduction
Provides a brief overview of some Vulkan considerations as well as a couple of notes on Rust.

[Introduction to Vulkan](doc_src/section_0.md)

## Our first window

For our initial project, we will open a black window. This terminally-boring example is actually much longer and more important than you might expect. It will introduce most of the critical aspects shared by any Vulkan program.

[First Window](doc_src/section_1.md)

## Rendering a triangle

With our first project we learned how to set up and use a Vulkan program. Now in this lesson we learn how to render things to the screen. This will involve writing our first shaders as well as passing information to them.

[Triangle](doc_src/section_2.md)

## Transformations

After getting your first triangle on the screen the obvious question becomes "great, so how do I make it do things?" For that we will need to learn to apply *transformations* via `Uniform` data, which will be the other main way we feed data to our shaders in addition to the vertex data we learned about in the last lesson.

[Transformations](doc_src/section_3.md)

## Depth

A shorter lesson that shows how to enable depth-testing in Vulkan. This is a necessity for the more complicated scenes we'll be creating in later lessons.

[Depth](doc_src/section_4.md)

## Winding order

This lesson is a short one but it talks about a very important subject for rendering more complicated scenes: winding order and how you can use it to lower the amount of work your graphics hardware has to do.

[Winding Order](doc_src/section_5.md)

## Light I

And in the sixth lesson the programmer said "let there be light" and lo, there was light!

[Light](doc_src/section_6.md)

## Multi-pass rendering

Now that we're starting to get into more advanced features we're going to need to increase the power of our rendering system. We will accomplish this by introducing multiple rendering passes into the equation.

[Multi-pass rendering](doc_src/section_7.md)

## Light II

Now that we've got our deferred rendering system set up, let's look at how you can use it to add more than one directional light to our scene.

[Multiple Lights](doc_src/section_8.md)

## Models

Now that we've set up a working lighting system let's take a step back and look at something else that would be important to any graphics application: how to load models from a file.

[Loading Models](doc_src/section_9.md)

## New Uniforms

After a couple of longer lessons, let's take a short break to refactor our normal code as well as set the stage for multiple models.

[Normal Uniform](doc_src/section_10.md)

## Render System I

We're ready now to take our first stab at making a real rendering system. We won't introduce any new concepts in this lesson, just wrap up what we already know in a way that lets us input models to renders as well as directional light sources. Think of this as a draft version of the more advanced system we'll make in a few lessons.

[Render System](doc_src/section_11.md)
