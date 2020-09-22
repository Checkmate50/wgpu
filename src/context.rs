use crate::shared::string_compare;

#[derive(Debug)]
pub struct BindingContext {
    starting_context: [&'static str; 32],
    result_context: [&'static str; 32],
    pub has_out_bound: bool,
    pub do_consume: bool,
}

impl BindingContext {
    pub const fn new(
        starting_context: [&'static str; 32],
        result_context: [&'static str; 32],
    ) -> BindingContext {
        BindingContext {
            starting_context,
            result_context,
            has_out_bound: false,
            do_consume: false,
        }
    }
}

pub const fn update_bind_context(
    bind_context: &BindingContext,
    bind_name: &'static str,
) -> BindingContext {
    let mut acc = 0;
    let mut found_it = false;
    let mut new_bind_context = [""; 32];

    let mut has_out_bound = bind_context.has_out_bound;
    let do_consume = bind_context.has_out_bound || has_out_bound;

    while acc < 32 {
        if string_compare(bind_context.starting_context[acc], bind_name) {
            found_it = true;
            if !has_out_bound && params_contain_string(&bind_context.result_context, bind_name) {
                has_out_bound = true;
            }
        } else {
            new_bind_context[acc] = bind_context.starting_context[acc];
        }
        acc += 1;
    }

    if !found_it {
        panic!("We did not find the parameter you are trying to bind to in the bind context")
    }

    BindingContext {
        starting_context: new_bind_context,
        result_context: bind_context.result_context,
        has_out_bound,
        do_consume,
    }
}

pub const fn ready_to_run(bind_context: BindingContext) {
    let mut acc = 0;

    while acc < 32 {
        if !string_compare(bind_context.starting_context[acc], "") {
            panic!("This bind context still has in parameters that need to be bound to")
        }
        acc += 1;
    }
}

const fn params_contain_string(list_of_names: &[&str; 32], name: &str) -> bool {
    let mut acc = 0;
    while acc < 32 {
        if string_compare(list_of_names[acc], name) {
            return true;
        }
        acc += 1;
    }
    false
}

pub const fn can_pipe(s_out: &BindingContext, s_in: &BindingContext) -> bool {
    let mut acc = 0;
    while acc < 32 {
        if !string_compare(s_out.result_context[acc], "")
            && !params_contain_string(&s_in.starting_context, s_out.result_context[acc])
        {
            return false;
        }
        acc += 1;
    }
    true
}
