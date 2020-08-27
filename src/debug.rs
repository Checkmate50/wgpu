//https://stackoverflow.com/questions/38141056/does-rust-have-a-debug-macro

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! debug {
    ($x:expr) => {
        dbg!(&$x)
    };
}

// for nice debug printing of strings
#[macro_export]
#[cfg(debug_assertions)]
macro_rules! debug_print {
    ($x:expr) => {{
        dbg!();
        println!("{}", $x)
    }};
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! debug {
    ($x:expr) => {
        std::convert::identity(&$x)
    };
}
