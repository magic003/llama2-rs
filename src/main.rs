use std::{
    env, process,
    time::{SystemTime, UNIX_EPOCH},
};

use llama2_rs::{Generator, Sampler, Tokenizer, Transformer};

const MODE_GENERATE: &str = "generate";
const MODE_CHAT: &str = "chat";

fn main() {
    // parse command line arguments and validate them
    let args: Vec<String> = env::args().collect();
    if args.len() % 2 != 0 {
        error_usage();
    }
    let checkpoint_path = &args[1];
    let mut temperature = 1.0f32; // 0.0 = greedy deterministic. 1.0 = original. Don't set higher
    let mut top_p = 0.9f32; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let mut random_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs(); // seed rng with time by default
    let mut steps = 256u32; // number of steps to run for
    let mut prompt = ""; // prompt string
    let mut tokenizer_path = "tokenizer.bin";
    let mut mode = MODE_GENERATE; // generate|chat
    let mut system_prompt = ""; // the (optional) system prompt to use in chat mode

    for opt in args.chunks_exact(2).skip(1) {
        let flag = &opt[0];
        let value = &opt[1];

        if flag.len() != 2 || !flag.starts_with('-') {
            error_usage();
        }
        if flag == "-t" {
            temperature = value
                .parse()
                .expect(&format!("Invalid temperature value: {}", value));
            if temperature < 0.0 {
                eprintln!("Temperature must be non-negative");
                process::exit(1);
            }
        } else if flag == "-p" {
            top_p = value
                .parse()
                .expect(&format!("Invalid top-p value: {}", value));
            if top_p < 0.0 || top_p > 1.0 {
                eprintln!("Top-p must be in the range [0, 1]");
                process::exit(1);
            }
        } else if flag == "-s" {
            random_seed = value
                .parse()
                .expect(&format!("Invalid random seed value: {}", value));
        } else if flag == "-n" {
            steps = value
                .parse()
                .expect(&format!("Invalid number of steps value: {}", value));
        } else if flag == "-i" {
            prompt = value;
        } else if flag == "-z" {
            tokenizer_path = value;
        } else if flag == "-m" {
            mode = value;
            if mode != MODE_GENERATE && mode != MODE_CHAT {
                eprintln!(
                    "Invalid mode: {}. Use '{}' or '{}'.",
                    mode, MODE_GENERATE, MODE_CHAT
                );
                process::exit(1);
            }
        } else if flag == "-y" {
            system_prompt = value;
        } else {
            error_usage();
        }
    }

    // build the Transformer via the model .bin file
    let transformer = Transformer::from_file(checkpoint_path)
        .expect(&format!("Failed to load model from {}", checkpoint_path));
    if steps == 0 || steps > transformer.config.seq_len {
        steps = transformer.config.seq_len;
    }

    // build the Tokenizer via the tokenizer .bin file
    let tokenizer = Tokenizer::new(&tokenizer_path, transformer.config.vocab_size)
        .expect(&format!("Failed to load tokenizer from {}", tokenizer_path));

    // build the Sampler
    let sampler = Sampler::new(
        transformer.config.vocab_size,
        temperature,
        if top_p == 1.0 { None } else { Some(top_p) },
        random_seed,
    );

    // build the Generator
    let mut generator = Generator::new(
        tokenizer,
        transformer,
        sampler,
        true, // report stats
    );

    // run!
    if mode == MODE_GENERATE {
        let output = generator.generate(prompt, steps);
        println!("{}\n", output);
    } else if mode == MODE_CHAT {
        // TODO
    }
}

fn error_usage() {
    eprint!("Usage:   run <checkpoint> [options]\n");
    eprint!("Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    eprint!("Options:\n");
    eprint!("  -t <float>  temperature in [0,inf], default 1.0\n");
    eprint!("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    eprint!("  -s <int>    random seed, default current Epoch time in seconds\n");
    eprint!("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    eprint!("  -i <string> input prompt\n");
    eprint!("  -z <string> optional path to custom tokenizer\n");
    eprint!("  -m <string> mode: generate|chat, default: generate\n");
    eprint!("  -y <string> (optional) system prompt in chat mode\n");
    process::exit(1);
}
