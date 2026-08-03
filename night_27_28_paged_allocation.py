import math
from typing import Dict, List

# Configuration
CONFIG = {
    "TOTAL_BLOCKS": 16,  # Total physical GPU blocks
    "BLOCK_SIZE": 4,  # Number of tokens per block (Page size)
}


class PagedAttentionMemoryManager:
    def __init__(self, total_blocks: int, block_size: int):
        # Pool of free physical block IDs
        self.free_blocks: List[int] = list(range(total_blocks))
        self.block_size: int = block_size

        # Block Tables: sequence_id -> list of physical block IDs
        self.block_tables: Dict[str, List[int]] = {}

        # Tracks current token counts per sequence
        self.seq_lengths: Dict[str, int] = {}

    def prefill(self, seq_id: str, prompt_len: int) -> bool:
        """Prefill Phase: Allocate physical blocks for the initial prompt at once."""
        needed_blocks = math.ceil(prompt_len / self.block_size)

        if len(self.free_blocks) < needed_blocks:
            print(
                f"❌ [Prefill OOM] Cannot allocate {needed_blocks} blocks for '{seq_id}'."
            )
            return False

        # Allocate blocks from the free stack
        allocated = [self.free_blocks.pop() for _ in range(needed_blocks)]
        self.block_tables[seq_id] = allocated
        self.seq_lengths[seq_id] = prompt_len

        print(
            f"🚀 [Prefill] '{seq_id}' (Prompt Len: {prompt_len}) -> Allocated Blocks: {allocated}"
        )
        return True

    def decode_step(self, seq_id: str) -> bool:
        """Decode Phase: Add 1 generated token. Dynamically fetch a new block when full."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not active.")

        current_len = self.seq_lengths[seq_id]

        # Check if current length exactly hits a block boundary
        # If so, the active block is full, and we need a new physical page for the next token.
        if current_len % self.block_size == 0:
            if not self.free_blocks:
                print(
                    f"❌ [Decode OOM] Memory full! Unable to allocate new block for '{seq_id}'."
                )
                return False

            new_block = self.free_blocks.pop()
            self.block_tables[seq_id].append(new_block)
            print(
                f"⚡ [Decode +1 Block] '{seq_id}' crossed boundary ({current_len} tokens) -> Allocated Physical Block {new_block}"
            )

        self.seq_lengths[seq_id] += 1
        return True

    def finish_sequence(self, seq_id: str) -> None:
        """Deallocate sequence memory, making physical blocks immediately re-usable."""
        if seq_id in self.block_tables:
            freed = self.block_tables.pop(seq_id)
            self.seq_lengths.pop(seq_id)
            self.free_blocks.extend(freed)
            print(
                f"🧹 [Deallocate] Finished '{seq_id}' -> Returned Blocks {freed} back to pool."
            )


# ==========================================
# Concurrent Prefill & Decode Simulation
# ==========================================

mem = PagedAttentionMemoryManager(
    total_blocks=CONFIG["TOTAL_BLOCKS"], block_size=CONFIG["BLOCK_SIZE"]
)

print(f"Initial Free Blocks ({len(mem.free_blocks)}): {mem.free_blocks}\n" + "=" * 60)

# --- Stage 1: Concurrent Prefills ---
# Sequence A prompt len: 6 (Needs ceil(6/4) = 2 blocks)
# Sequence B prompt len: 3 (Needs ceil(3/4) = 1 block)
mem.prefill("Seq_A", prompt_len=6)
mem.prefill("Seq_B", prompt_len=3)

print(f"\nRemaining Free Blocks: {mem.free_blocks}\n" + "=" * 60)

# --- Stage 2: Interleaved Decode Generation ---
print("--- Starting Decode Iterations ---\n")

# Step 1: Decode Seq_A (Length goes 6 -> 7; fills active block, no new allocation needed)
mem.decode_step("Seq_A")

# Step 2: Decode Seq_B (Length goes 3 -> 4; fills block 0-3 capacity)
mem.decode_step("Seq_B")

# Step 3: Decode Seq_B again (Length 4 hits block boundary -> Triggers dynamic allocation of new block)
mem.decode_step("Seq_B")

# Step 4: Decode Seq_A (Length 7 -> 8 hits block boundary -> Triggers dynamic allocation)
mem.decode_step("Seq_A")

print(f"\nActive Block Tables: {mem.block_tables}")
print(f"Remaining Free Blocks: {mem.free_blocks}\n" + "=" * 60)

# --- Stage 3: Deallocation ---
# Seq_A finishes generation and releases memory
mem.finish_sequence("Seq_A")

# Seq_C pre-fills using reclaimed memory
mem.prefill("Seq_C", prompt_len=7)
