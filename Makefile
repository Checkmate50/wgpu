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

.PHONY: texture
texture:
	#export RUST_BACKTRACE=1 && cargo +nightly run --example skybox_texture
	cargo +nightly run --example skybox_texture

.PHONY: teapot
teapot:
	cargo +nightly run --example teapot

.PHONY: cube
cube:
	cargo +nightly run --example cube
	#cargo +nightly run --example cube --release

.PHONY: cube2
cube2:
	export RUST_BACKTRACE=1 && cargo +nightly run --example cube_texture
	#cargo +nightly run --example cube_texture --release

.PHONY: cube3
cube3:
	cargo +nightly run --example cube_shadow
	#cargo +nightly run --example cube_shadow --release

.PHONY: cube4
cube4:
	cargo +nightly run --example multicube
	#cargo +nightly run --example cube_shadow --release

.PHONY: flat_color
flat_color:
	cargo +nightly run --example flat_color

.PHONY: shadow
shadow:
	cargo +nightly run --example shadow

.PHONY: sink
sink:
	cargo +nightly run --example sink

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