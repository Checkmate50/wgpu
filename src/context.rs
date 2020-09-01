use crate::shared::string_compare;

#[derive(Debug)]
pub struct MetaContext {
    mutating_contexts: [&'static str; 32],
    mut_acc : u32,
    invalidated_contexts: [&'static str; 32],
    inval_acc : u32,
}

impl MetaContext {
    pub const fn new() -> MetaContext {
        MetaContext {
            mutating_contexts: [""; 32],
            mut_acc : 0,
            invalidated_contexts: [""; 32],
            inval_acc : 0,
        }
    }

    pub const fn is_valid(&self, bc: &BindingContext) -> bool {
        let mut acc = 0;
        while acc < self.inval_acc {
            if string_compare(self.invalidated_contexts[acc as usize], bc.name) {
                return false;
            }
            acc += 1;
        }
        true
    }

    pub const fn update(&self, bc: &BindingContext) -> MetaContext {
        let mut mutating_contexts = self.mutating_contexts;
        mutating_contexts[self.mut_acc as usize] = bc.name;
        let mut_acc = self.mut_acc + 1;
        MetaContext {
            mutating_contexts,
            mut_acc,
            invalidated_contexts : self.invalidated_contexts,
            inval_acc : self.inval_acc,
        }
    }

    pub const fn invalidate_context(&self) -> MetaContext {
        let mut acc = 0;
        let mut invalidated_contexts = self.invalidated_contexts;
        let mut inval_acc = self.inval_acc;
        while acc < self.mut_acc {
            if inval_acc < 32 {
            invalidated_contexts[inval_acc as usize] = self.mutating_contexts[acc as usize];
            inval_acc += 1;
            } else {
                panic!("whoops we ran out of space in out meta-context and idk what to do")
            }
            acc += 1;
        }
        MetaContext {
            mutating_contexts : [""; 32],
            mut_acc : 0,
            invalidated_contexts,
            inval_acc,
        }
    }
}

#[derive(Debug)]
pub struct BindingContext {
    starting_context: [&'static str; 32],
    result_context: [&'static str; 32],
    name: &'static str,
    has_out_bound: bool,
}

impl BindingContext {
    pub const fn new(
        starting_context: [&'static str; 32],
        result_context: [&'static str; 32],
    ) -> BindingContext {
        BindingContext {
            starting_context,
            result_context,
            name: "",
            has_out_bound: false,
        }
    }
}

pub const fn update_bind_context(
    bind_context: &BindingContext,
    bind_name: &'static str,
    meta_context: MetaContext,
    context_name : &'static str,
) -> (BindingContext, MetaContext) {
    let mut acc = 0;
    let mut found_it = false;
    let mut new_bind_context = [""; 32];

    let mut has_out = bind_context.has_out_bound;

    if !meta_context.is_valid(bind_context) {
        panic!("This bind context has been invalidated")
    }

    while acc < 32 {
        if string_compare(bind_context.starting_context[acc], bind_name) {
            found_it = true;
            if !has_out && params_contain_string(&bind_context.result_context, bind_name) {
                has_out = true;
            }
        } else {
            new_bind_context[acc] = bind_context.starting_context[acc];
        }
        acc += 1;
    }

    if !found_it {
        panic!("We did not find the parameter you are trying to bind to in the bind context")
    }

    let bc = BindingContext {
        starting_context: new_bind_context,
        result_context: bind_context.result_context,
        has_out_bound: has_out,
        name : context_name,
    };

    let new_meta_context = if has_out {
        MetaContext::update(&meta_context, &bc)
    } else {
        meta_context
    };

    (bc, new_meta_context)
}

pub const fn ready_to_run(bind_context: BindingContext, meta_context: MetaContext) -> MetaContext {
    let mut acc = 0;

    if !meta_context.is_valid(&bind_context) {
        panic!("This bind context has been invalidated");
    }

    while acc < 32 {
        if !string_compare(bind_context.starting_context[acc], "") {
            panic!("This bind context still has in parameters that need to be bound to")
        }
        acc += 1;
    }
    MetaContext::invalidate_context(&meta_context)
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
