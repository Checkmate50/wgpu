.PHONY: trivial
trivial:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_compute
	cargo +nightly run --example trivial_compute

.PHONY: pipeline
pipeline:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_pipeline
	cargo +nightly run --example trivial_pipeline

.PHONY: hello
hello: hello
	#export RUST_BACKTRACE=1 && cargo +nightly run --example hello_compute
	cargo +nightly run --example hello_compute

.PHONY: boids
boids: boids
	#export RUST_BACKTRACE=1 && cargo +nightly run --example boids_compute
	cargo +nightly run --example boids_compute

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