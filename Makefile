all:
	build

build:
	cargo build

run: build
	export RUST_BACKTRACE=1 && cargo run

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