//use wgpu_macros::my_macro;

/* my_macro!{var1, var2} */

use wgpu_macros::generic_bindings;

// [in out] var1
// [in] var2
generic_bindings! {var1, var2, var1}

/* trait AbstractBind {
    fn new() -> Self;
}

struct Bound {}

struct Unbound {}

impl AbstractBind for Bound {
    fn new() -> Self {
        Bound {}
    }
}

impl AbstractBind for Unbound {
    fn new() -> Self {
        Unbound {}
    }
} */

/*
    trait AbstractContext {
        fn new() -> Self;
    }

    #[derive(Clone)]
    struct Context {}

    #[derive(Clone)]
    struct MutContext {}

    impl AbstractContext for Context {
        fn new() -> Self {
            Context {}
        }
    }

    impl AbstractContext for MutContext {
        fn new() -> Self {
            MutContext {}
        }
    }

    fn init() -> (Unbound, Unbound, Context) {
        (Unbound {}, Unbound {}, Context {})
    }

    fn bind1<R: AbstractBind + Clone>(a: &(Unbound, R, Context)) -> (Bound, R, Context) {
        let (_, b, _) = a;
        (Bound {}, b.clone(), Context {})
    }

    fn bind1_consume<R: AbstractBind + Clone>(a: (Unbound, R, MutContext)) -> (Bound, R, MutContext) {
        let (_, b, _) = a;
        (Bound {}, b.clone(), MutContext {})
    }

    fn bind2_mutate<R: AbstractBind + Clone>(a: &(R, Unbound, Context)) -> (R, Bound, MutContext) {
        let (b, _, _) = a;
        (b.clone(), Bound {}, MutContext {})
    }

    fn bind2_consume<R: AbstractBind + Clone>(a: (R, Unbound, MutContext)) -> (R, Bound, MutContext) {
        let (b, _, _) = a;
        (b.clone(), Bound {}, MutContext {})
    }

    fn run<A : AbstractContext>(a: (Bound, Bound, A)) {
        println!("hello")
    }
*/

/* fn init() -> (Unbound, Unbound) {
    (Unbound {}, Unbound {})
} */

/* fn bind_var1<R: AbstractBind>(_: &(Unbound, R)) -> (Bound, R) {
    (Bound::new(), R::new())
}

fn bind_var2<R: AbstractBind>(_: &(R, Unbound)) -> (R, Bound) {
    (R::new(), Bound::new())
} */

/* fn run(a: (Bound, Bound)) {
    println!("hello")
} */

fn main() {
    let s = init();
    {
        let y = bind_var2(&s);
        {
            let t = bind_var1_mutate(&y);
            {
                run(t);
            }
        }
        /*         let t = bind_var1_mutate(&s);
        {
            let y = bind_var2_consume(t);
            {
                run(y);
            }
        } */
    }
}

/* fn main() {
    let s : ShaderContext_var1_var2 = ShaderContext_var1_var2 {};
    let t : ShaderContext_var2 = s.bind_var1();
    let u : ShaderContext = t.bind_var2();
    u.run()
} */
