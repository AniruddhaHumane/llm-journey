import math
from typing import Dict, List, Optional

CONFIG = {
    "TOTAL_BLOCKS": 16,  # Total physical GPU blocks available
    "BLOCK_SIZE": 4,  # Number of tokens per block
}


class DynamicContiguousMemoryManager:
    def __init__(self, total_blocks: int, block_size: int):
        self.total_blocks = total_blocks
        self.block_size = block_size

        # Physical memory pool tracked as an array (True = Free, False = Allocated)
        self.free_pool: List[bool] = [True] * total_blocks

        # Block Tables: sequence_id -> list of contiguous physical block IDs
        self.block_tables: Dict[str, List[int]] = {}
        self.seq_lengths: Dict[str, int] = {}
        self.req_max_lens: Dict[str, int] = {}

    def _find_contiguous_blocks(self, count: int) -> Optional[int]:
        """Finds the starting block index for 'count' contiguous free blocks."""
        consecutive = 0
        for i in range(self.total_blocks):
            if self.free_pool[i]:
                consecutive += 1
                if consecutive == count:
                    return i - count + 1
            else:
                consecutive = 0
        return None

    def prefill(self, seq_id: str, prompt_len: int, req_max_len: int) -> bool:
        """Prefill Phase: Pre-allocates memory for a per-request maximum length contiguously."""
        if prompt_len > req_max_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) cannot exceed req_max_len ({req_max_len})"
            )

        # Calculate blocks needed based on per-request max length
        needed_blocks = math.ceil(req_max_len / self.block_size)

        start_idx = self._find_contiguous_blocks(needed_blocks)
        if start_idx is None:
            total_free = sum(self.free_pool)
            print(
                f"❌ [{seq_id}] PREFILL OOM: Requested {needed_blocks} contiguous blocks (req_max_len={req_max_len}). "
                f"Total free blocks available: {total_free}, but NO contiguous run of size {needed_blocks} exists!"
            )
            return False

        # Allocate contiguous slice
        allocated_blocks = list(range(start_idx, start_idx + needed_blocks))
        for b in allocated_blocks:
            self.free_pool[b] = False

        self.block_tables[seq_id] = allocated_blocks
        self.seq_lengths[seq_id] = prompt_len
        self.req_max_lens[seq_id] = req_max_len

        print(
            f"🚀 [{seq_id}] PREFILL Success | Prompt: {prompt_len}, MaxLen: {req_max_len} "
            f"-> Allocated Contiguous Blocks {allocated_blocks}"
        )
        return True

    def decode_step(self, seq_id: str) -> bool:
        """Decode Phase: Advance token count within pre-reserved contiguous range."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not active.")

        current_len = self.seq_lengths[seq_id]
        req_max_len = self.req_max_lens[seq_id]

        if current_len >= req_max_len:
            print(
                f"❌ [{seq_id}] DECODE Error: Reached per-request max length ({req_max_len} tokens)."
            )
            return False

        self.seq_lengths[seq_id] += 1
        print(
            f"⚡ [{seq_id}] DECODE Token {self.seq_lengths[seq_id]}/{req_max_len} (No memory allocation needed)"
        )
        return True

    def finish_sequence(self, seq_id: str) -> None:
        """Deallocate sequence memory, creating potential holes in physical memory."""
        if seq_id in self.block_tables:
            freed = self.block_tables.pop(seq_id)
            self.seq_lengths.pop(seq_id)
            self.req_max_lens.pop(seq_id)

            for b in freed:
                self.free_pool[b] = True

            print(f"🧹 [{seq_id}] FINISHED -> Freed Blocks {freed}")

    def print_memory_map(self) -> None:
        """Helper to visualize the physical memory array."""
        mem_map = []
        for i in range(self.total_blocks):
            owner = "."
            for seq_id, blocks in self.block_tables.items():
                if i in blocks:
                    owner = seq_id[-1]  # Get letter (A, B, C...)
                    break
            mem_map.append(f"[{owner}]")
        print("Memory Map: " + "".join(mem_map))


# ============================================================
# External Fragmentation OOM Scenario
# ============================================================

mem = DynamicContiguousMemoryManager(
    total_blocks=CONFIG["TOTAL_BLOCKS"],
    block_size=CONFIG["BLOCK_SIZE"],
)

print("=" * 70)
print("--- Step 1: Allocate mixed sizes to fill all 16 blocks ---")
# req_max_len of 16 -> 4 blocks, 8 -> 2 blocks, 12 -> 3 blocks, etc.
mem.prefill("Seq_A", prompt_len=4, req_max_len=16)  # 4 blocks -> [0, 1, 2, 3]
mem.prefill("Seq_B", prompt_len=2, req_max_len=8)  # 2 blocks -> [4, 5]
mem.prefill("Seq_C", prompt_len=4, req_max_len=16)  # 4 blocks -> [6, 7, 8, 9]
mem.prefill("Seq_D", prompt_len=2, req_max_len=8)  # 2 blocks -> [10, 11]
mem.prefill("Seq_E", prompt_len=4, req_max_len=16)  # 4 blocks -> [12, 13, 14, 15]

mem.print_memory_map()

print("\n" + "=" * 70)
print("--- Step 2: Finish non-adjacent small sequences (Seq_B and Seq_D) ---")
mem.finish_sequence("Seq_B")  # Frees blocks [4, 5]
mem.finish_sequence("Seq_D")  # Frees blocks [10, 11]

mem.print_memory_map()

print("\n" + "=" * 70)
print("--- Step 3: Request Seq_F needing 3 contiguous blocks (12 tokens) ---")
# Total free blocks = 4 (blocks 4, 5, 10, 11)
# Required contiguous blocks = 3 (for req_max_len = 12)
# Result: External Fragmentation OOM!
mem.prefill("Seq_F", prompt_len=3, req_max_len=12)
