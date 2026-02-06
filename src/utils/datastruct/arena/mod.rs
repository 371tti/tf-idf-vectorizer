#[derive(Debug, Clone)]
pub struct Arena<T> {
    pool: Vec<Entry<T>>,
    free_list: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum Entry<T> {
    None {
        before_gen_id: u32,
    },
    Some {
        value: T,
        gen_id: u32,
    },
}

impl<T> Entry<T> {
    fn new_entry(&mut self, value: T) {
        match self {
            Entry::None { before_gen_id } => {
                *self = Entry::Some {
                    value,
                    gen_id: *before_gen_id + 1,
                };
            }
            Entry::Some { .. } => panic!("Inconsistent state: trying to allocate on occupied entry"),
        }
    }

    fn new(value: T) -> Self {
        Entry::Some { value, gen_id: 0 }
    }

    fn get_gen_id(&self) -> u32 {
        match self {
            Entry::None { before_gen_id } => *before_gen_id,
            Entry::Some { gen_id, .. } => *gen_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaSlot {
    idx: u32,
    gen_id: u32,
}

impl ArenaSlot {
    fn new(index: usize, gen_id: u32) -> Self {
        Self {
            idx: index as u32,
            gen_id,
        }
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn alloc(&mut self, value: T) -> ArenaSlot {
        if let Some(free_idx) = self.free_list.pop() {
            let entry = &mut self.pool[free_idx];
            entry.new_entry(value);
            ArenaSlot::new(free_idx, entry.get_gen_id())
        } else {
            let entry = Entry::new(value);
            self.pool.push(entry);
            let idx = self.pool.len() - 1;
            ArenaSlot::new(idx, self.pool[idx].get_gen_id())
        }
    }

    pub fn dealloc(&mut self, slot: &ArenaSlot) {
        let entry = &mut self.pool[slot.idx as usize];
        match entry {
            Entry::Some { gen_id, .. } if *gen_id == slot.gen_id => {
                let gen_id = entry.get_gen_id();
                *entry = Entry::None {
                    before_gen_id: gen_id,
                };
                self.free_list.push(slot.idx as usize);
            }
            _ => {
                // slot is invalid or already freed
            }
        }
    }

    pub fn get(&self, slot: &ArenaSlot) -> Option<&T> {
        let entry = &self.pool[slot.idx as usize];
        match entry {
            Entry::Some { value, gen_id } if *gen_id == slot.gen_id => Some(value),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, slot: &ArenaSlot) -> Option<&mut T> {
        let entry = &mut self.pool[slot.idx as usize];
        match entry {
            Entry::Some { value, gen_id } if *gen_id == slot.gen_id => Some(value),
            _ => None,
        }
    }
}