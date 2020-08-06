.PHONY: trivial
trivial:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_compute
	cargo +nightly run --example trivial_compute

.PHONY: pipeline
pipeline:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example trivial_pipeline
	cargo +nightly run --example trivial_pipeline

.PHONY: hello
hello:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example hello_compute
	cargo +nightly run  --example hello_compute

.PHONY: boids
boids:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example boids_compute
	cargo +nightly run --example boids_compute

.PHONY: triangle
triangle:
	export RUST_BACKTRACE=1 && cargo +nightly run --example hello_triangle
	#cargo +nightly run --example hello_triangle

.PHONY: boids2
boids2:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example boids_graphics
	cargo +nightly run --example boids_graphics

.PHONY: boids3
boids3:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example boids_image
	cargo +nightly run --example boids_image

.PHONY: texture
texture:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example skybox_texture
	cargo +nightly run --example skybox_texture

.PHONY: teacup
teacup:
	cargo +nightly run --example teacup

release:
	cargo build --release

test:
	cargo test

clean:
	cargo clean

check:
	cargo +nightly clippy

format:
	cargo +nightly fmt