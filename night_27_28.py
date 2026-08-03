from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
import random
import string


config = {
    "TOTAL_BLOCKS": 64,  # Total GPU physical blocks available
    "BLOCK_SIZE": 16,  # Tokens per block (e.g., 16 tokens/block)
    "B_max": 8,  # Max concurrent requests in batch
    "MAX_LEN": 256,  # Max possible sequence length (for naive pre-allocation)
}
chars = string.ascii_letters + string.digits


initially_allocated_blocks = random.randint(0, config["TOTAL_BLOCKS"] // 2)

free_mmap = [
    [initially_allocated_blocks, config["TOTAL_BLOCKS"] - 1]
]  # shows which blocks are free
allocated_mmap = {-(i + 1) * 10: (i, i) for i in range(initially_allocated_blocks)}


n_blocks = config["MAX_LEN"] // config["BLOCK_SIZE"]  # number of blocks
total_seq = (
    config["TOTAL_BLOCKS"] / n_blocks
)  # total number of sequence that can fit at a time if allocation was perfect

random_block_chance = 0.5  # this controls if memory blocks get allocated for things other than sequence to showcase fragmentation


def allocate_random_blocks(index: int):
    if len(free_mmap) > 0:
        first_free_slot = free_mmap[0]
        allocated_mmap[-index] = (first_free_slot[0], first_free_slot[0])
        if first_free_slot[1] == first_free_slot[0]:
            del free_mmap[0]
        else:
            first_free_slot[0] = first_free_slot[0] + 1


def deallocate_random_blocks():
    blocks_to_delete = random.randint(0, 10)
    console.log(f"deleting {blocks_to_delete} random blocks... (if exist)")
    for i in range(blocks_to_delete):
        idx_to_delete = None
        for block_key, _ in allocated_mmap.items():
            if block_key < 0:
                idx_to_delete = block_key
                break
        if idx_to_delete is not None:
            free_mmap.append(
                [allocated_mmap[idx_to_delete][0], allocated_mmap[idx_to_delete][0]]
            )
            del allocated_mmap[idx_to_delete]


def static_allocator(seq: int):
    global free_mmap
    deletion_plan = [False] * len(free_mmap)

    # all sequences are assumed to be MAX_LEN tokens or n_blocks blocks
    if seq in allocated_mmap:
        return False
    memory_exists = False
    for i, free_mem in enumerate(free_mmap):
        if free_mem[1] + 1 - free_mem[0] >= n_blocks:
            memory_exists = True
            break

    if memory_exists:
        allocated_mmap[seq] = (free_mem[0], free_mem[0] + n_blocks - 1)
        if free_mem[1] + 1 - free_mem[0] == n_blocks:
            deletion_plan[i] = True
        else:
            free_mem[0] = free_mem[0] + n_blocks

        console.log(f"Allocated: {seq, allocated_mmap[seq][0], allocated_mmap[seq][1]}")
        free_mmap = [mem for i, mem in enumerate(free_mmap) if not deletion_plan[i]]

        random_chance = random.random()
        # console.log(f"random allocation chance: {random_chance}")
        if random_chance < random_block_chance:
            allocate_random_blocks(seq)

        return seq, allocated_mmap[seq][0], allocated_mmap[seq][1]
    else:
        console.log("Memory insufficient!")
        return False


def merge_free_mmap():
    global free_mmap
    if len(free_mmap) > 1:
        mmap = sorted(free_mmap)
        del_map = [False] * len(free_mmap)
        for i in range(len(mmap) - 1):
            if mmap[i][1] + 1 == mmap[i + 1][0]:
                mmap[i + 1][0] = mmap[i][0]
                del_map[i] = True
        free_mmap = [mem for i, mem in enumerate(mmap) if not del_map[i]]


def static_deallocator(seq: int):
    global allocated_mmap
    global free_mmap

    if seq in allocated_mmap:
        start, end = allocated_mmap[seq]
        del allocated_mmap[seq]
        free_mmap.append([start, end])
        random_chance = random.random()
        console.log(f"random deallocation chance: {random_chance}")
        if random_chance < random_block_chance:
            deallocate_random_blocks()
        merge_free_mmap()
        console.log(f"deallocated: {seq, start, end}")
        return True
    console.log("Seq not found!")
    return False


free_mmap = {i for i in range(config["TOTAL_BLOCKS"])}
allocated_mmap = {}

# seq = "".join(random.choices(chars, k=4))


def paged_allocator(seq: str, seq_len: int = n_blocks) -> bool:
    if len(free_mmap) < seq_len:
        return False

    # Fast O(K) allocation via stack pops
    allocated_mmap[seq] = [free_mmap.pop() for _ in range(seq_len)]
    return True


def paged_deallocator(seq: str) -> None:
    if seq in allocated_mmap:
        # Fast O(K) reclamation via stack extend
        free_mmap.extend(allocated_mmap[seq])
        del allocated_mmap[seq]


console = Console()
SEQ_COLORS = [
    "cyan",
    "magenta",
    "yellow",
    "green",
    "red",
    "blue",
    "bright_blue",
    "bright_magenta",
]


# ---------------------------------------------------------
# Rich Visualization Renderer
# ---------------------------------------------------------
def render_memory_grid():
    total_blocks = config["TOTAL_BLOCKS"]
    grid_data = [None] * total_blocks

    # Populate grid with allocated blocks
    for seq_id, (start, end) in allocated_mmap.items():
        color = SEQ_COLORS[seq_id % len(SEQ_COLORS)]
        for b in range(start, end + 1):
            grid_data[b] = (seq_id, color)

    # 1. Build Memory Grid Table (8x8)
    grid_table = Table(
        title="[bold white]GPU Physical Memory Blocks (64 Total)[/bold white]",
        show_header=True,
        header_style="bold dim white",
        border_style="bright_black",
        expand=True,
    )

    cols = 8
    for i in range(cols):
        grid_table.add_column(f"Col {i}", justify="center")

    for row_idx in range(0, total_blocks, cols):
        row_cells = []
        for col_idx in range(cols):
            block_idx = row_idx + col_idx
            block_info = grid_data[block_idx]

            if block_info is None:
                # Free Block
                cell_text = Text()
                cell_text.append(f"B{block_idx:02d}\n", style="dim white")
                cell_text.append("[FREE]", style="bold green")
            else:
                # Allocated Block
                seq_id, color = block_info
                cell_text = Text()
                cell_text.append(f"B{block_idx:02d}\n", style="dim white")
                cell_text.append(f"S_{seq_id}", style=f"bold {color}")

            row_cells.append(cell_text)
        grid_table.add_row(*row_cells)

    # 2. Build Free Memory Ranges Panel
    free_text = Text()
    if not free_mmap:
        free_text.append("No free blocks available!", style="bold red")
    else:
        for r in free_mmap:
            size = r[1] - r[0] + 1
            free_text.append(f"• Blocks [{r[0]:02d} - {r[1]:02d}] ", style="bold green")
            free_text.append(f"({size} blocks free)\n", style="dim")

    free_panel = Panel(
        free_text,
        title="[bold green]Free Memory Ranges (free_mmap)[/bold green]",
        border_style="green",
    )

    # 3. Build Sequence Allocation Map Panel
    alloc_text = Text()
    if not allocated_mmap:
        alloc_text.append("No active allocations.", style="dim")
    else:
        for seq_id, (start, end) in sorted(allocated_mmap.items()):
            color = SEQ_COLORS[seq_id % len(SEQ_COLORS)]
            alloc_text.append(
                f"• {'Sequence' if int(seq_id) >= 0 else 'block'} {seq_id}: ",
                style=f"bold {color}",
            )
            alloc_text.append(f"Blocks [{start:02d} - {end:02d}]\n", style="white")

    alloc_panel = Panel(
        alloc_text,
        title="[bold cyan]Allocated Sequences (allocated_mmap)[/bold cyan]",
        border_style="cyan",
    )

    # Combine layout
    side_panels = Columns([free_panel, alloc_panel], expand=True)

    console.print("\n")
    console.print(Panel(grid_table, border_style="bright_blue"))
    console.print(side_panels)
