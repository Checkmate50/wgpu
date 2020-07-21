.PHONY: trivial
trivial:
	export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_compute

.PHONY: trivial2
trivial2:
	export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_2_func_compute

.PHONY: pipeline
pipeline:
	export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_pipeline

.PHONY: hello
hello: hello
	export RUST_BACKTRACE=1 && cargo +nightly run --example hello_compute

release:
	cargo build --release

test:
	cargo test

clean:
	cargo clean

check:
	cargo clippy

format:
	cargo fmt