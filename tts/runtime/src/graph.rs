use std::collections::{BTreeMap, HashMap};

use crate::ops::{self, Args, Kernel};
use crate::proto;
use crate::tensor::Tensor;

/// Named tensors, the currency between stages: each graph picks its inputs by name out of everything known so far.
pub type Values = BTreeMap<String, Tensor>;

/// One exported stage, with everything that depends only on its weights already evaluated.
pub struct Graph {
    inputs: Vec<(String, usize)>,
    outputs: Vec<(String, usize)>,
    constants: Vec<(usize, Tensor)>,
    steps: Vec<Step>,
    slots: usize,
}

struct Step {
    op: String,
    kernel: Kernel,
    inputs: Vec<Option<usize>>,
    /// Whether this is the input's last read, so the kernel may take (and reuse) its buffer.
    last: Vec<bool>,
    outputs: Vec<usize>,
}

impl Graph {
    pub fn load(bytes: &[u8]) -> Result<Graph, String> {
        let model = proto::parse(bytes)?;
        let mut ids: HashMap<String, usize> = HashMap::new();
        fn id(ids: &mut HashMap<String, usize>, name: &str) -> usize {
            let n = ids.len();
            *ids.entry(name.to_string()).or_insert(n)
        }
        let mut known: HashMap<usize, Tensor> = HashMap::new();
        for (name, t) in model.initializers {
            known.insert(id(&mut ids, &name), t);
        }
        // Older exporters also list initializers as graph inputs.
        let fed: Vec<String> = model
            .inputs
            .into_iter()
            .filter(|n| !ids.contains_key(n))
            .collect();
        let inputs = fed
            .into_iter()
            .map(|n| (id(&mut ids, &n), n))
            .map(|(i, n)| (n, i))
            .collect();
        let mut steps = vec![];
        for node in &model.nodes {
            let slots: Vec<Option<usize>> = node
                .inputs
                .iter()
                .map(|n| (!n.is_empty()).then(|| id(&mut ids, n)))
                .collect();
            let outputs: Vec<usize> = node.outputs.iter().map(|n| id(&mut ids, n)).collect();
            let constant: Vec<Option<&Tensor>> = slots
                .iter()
                .map(|s| s.and_then(|s| known.get(&s)))
                .collect();
            let (kernel, captured) =
                ops::prepare(node, &constant).map_err(|e| format!("{}: {e}", node.op))?;
            if slots
                .iter()
                .zip(&constant)
                .all(|(s, c)| s.is_none() || c.is_some())
            {
                let folded = kernel(Args(constant.iter().map(|c| c.cloned()).collect()));
                known.extend(outputs.into_iter().zip(folded));
                continue;
            }
            // A kernel that captured its constant operands (packed weights) doesn't need them passed in.
            let inputs = slots
                .iter()
                .zip(&constant)
                .enumerate()
                .map(|(i, (s, c))| {
                    if captured && i > 0 && c.is_some() {
                        None
                    } else {
                        *s
                    }
                })
                .collect();
            steps.push(Step {
                op: node.op.clone(),
                kernel,
                inputs,
                last: vec![],
                outputs,
            });
        }
        let outputs: Vec<_> = model.outputs.iter().map(|n| (n.clone(), ids[n])).collect();
        // Free each value after its last read; graph outputs live to the end.
        let mut seen: Vec<bool> = vec![false; ids.len()];
        for (_, slot) in &outputs {
            seen[*slot] = true;
        }
        for step in steps.iter_mut().rev() {
            // Reversed, so an input read twice by one node is taken by its later occurrence.
            step.last = step
                .inputs
                .iter()
                .rev()
                .map(|s| s.is_some_and(|s| !std::mem::replace(&mut seen[s], true)))
                .collect();
            step.last.reverse();
        }
        let constants = known.into_iter().filter(|(slot, _)| seen[*slot]).collect();
        Ok(Graph {
            inputs,
            outputs,
            constants,
            steps,
            slots: ids.len(),
        })
    }

    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(|(n, _)| n.as_str())
    }

    /// Runs on whichever of `values` the graph takes as inputs.
    pub fn run(&self, values: &Values) -> Values {
        self.run_with(values, &mut |_, kernel| kernel())
    }

    /// Like `run`, with every kernel call made through `hook(op, kernel)` (for profiling).
    pub fn run_with(
        &self,
        values: &Values,
        hook: &mut dyn FnMut(&str, &mut dyn FnMut()),
    ) -> Values {
        let mut slots: Vec<Option<Tensor>> = vec![None; self.slots];
        for (slot, t) in &self.constants {
            slots[*slot] = Some(t.clone());
        }
        for (name, slot) in &self.inputs {
            slots[*slot] = Some(
                values
                    .get(name)
                    .unwrap_or_else(|| panic!("missing input {name}"))
                    .clone(),
            );
        }
        for step in &self.steps {
            let mut args = Some(Args(
                step.inputs
                    .iter()
                    .zip(&step.last)
                    .map(|(s, &last)| {
                        s.and_then(|s| {
                            if last {
                                slots[s].take()
                            } else {
                                slots[s].clone()
                            }
                        })
                    })
                    .collect(),
            ));
            let mut out = vec![];
            hook(&step.op, &mut || out = (step.kernel)(args.take().unwrap()));
            for (slot, t) in step.outputs.iter().zip(out) {
                slots[*slot] = Some(t);
            }
        }
        self.outputs
            .iter()
            .map(|(n, s)| (n.clone(), slots[*s].take().expect("output computed")))
            .collect()
    }
}
