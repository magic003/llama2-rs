# llama2-rs

Inference Llama 2 in Rust. It ports the [llama2.c](https://github.com/karpathy/llama2.c) to Rust. I wrote this purely for
self educational purpose. While the code is pretty much replicated from [llama2.c](https://github.com/karpathy/llama2.c), 
it divides the code into several modules.

## Build and Test

To run the unit tests:
```bash
% cargo test
```

To make the release build:
```bash
% cargo build --release
```

## Run

It runs with the TinyStories [models](https://huggingface.co/karpathy/tinyllamas). I didn't test it with any Llama2 model though.

Running with the stories260K model:
```bash
./target/release/llama2-rs stories260K/stories260K.bin -z stories260K/tok512.bin -t 0.9 -s 12345 
```

It gives:
```
acheived tok/s 371.5529753265602

Once upon a time, there was a little girl named Lily. She loved to play with her toys in her room. One day, she asked her mom what happened. Her mom said, "Don't worry, Lily. We can go to the park to make it clean and shoot a new ones."
Lily was so happy and started to play all day. She asked her mom to help the party. Her mom said, "Let's find my bed." Lily and her mom sat down to the park, but it was not broken.
Lily's mom asked her what was wrong and they had a lot of fun. Lily smiled and said, "Okay, Lily. Let's see your mom."
Lily went to the park and got it in her bed. She showed it to her
```
With the same seed, it generates identical text as llama2.c.

You can download other TinyStories [models](https://huggingface.co/karpathy/tinyllamas), run:
```bash
./target/release/llama2-rs ~/Downloads/stories110M.bin
```

## Performance

Though better performance is not the goal, I did a little bit comparison with llama2.c just to validate the implementation.

The benchmark ran on a Mac Mini with the M2 chip. For llama2-rs, it uses the release version. For llama2.c, it uses the
build from `make run` (not `make runfast` which runs much faster), without OpenMP. They have similar performance.

This repo is the multi-thread version of llama2-rs. I also used a single thread version during the bechmark.

| model | llama2.c | llama2-rs (single thread) | llama2-rs (multi-thread) |
| --- | --- | --- | --- |
| [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | 123 tok/s | 125 tok/s | 190 tok/s |
| [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) | 35 tok/s | 36 tok/s | 89 tok/s |
| [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) | 12 tok/s | 12 tok/s | 38 tok/s |

## Unsorted TODOs

* Chat
* Quantization

## License

MIT
