from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    Represents a single block in the KV cache that stores token IDs and a hash value.
    
    Each block can be referenced by multiple sequences, and the ref_count tracks
    how many sequences are using this block. When ref_count reaches zero, the block
    can be recycled.
    
    Attributes:
        block_id (int): The unique identifier for this block.
        ref_count (int): Number of sequences currently referencing this block.
        hash (int): Hash value of the token sequence stored in this block (or -1 if not hashed).
        token_ids (list[int]): The actual token IDs stored in this block.
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """Update the block's hash value and token IDs."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Reset the block for new allocation with a ref_count of 1."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    Manages the allocation and deallocation of blocks for storing token sequences.
    
    The BlockManager uses a hashing strategy to efficiently reuse blocks that contain
    identical token sequences, enabling sharing of KV cache blocks between sequences
    that have common prefixes.
    
    This implementation handles various edge cases related to sequence lengths and
    block sizes, ensuring that blocks are properly allocated even when sequences
    grow or shrink in unexpected ways.
    
    Attributes:
        block_size (int): Number of tokens that can be stored in a single block.
        blocks (list[Block]): List of all blocks managed by this BlockManager.
        hash_to_block_id (dict[int, int]): Mapping from hash values to block IDs for quick lookup.
        free_block_ids (deque[int]): Queue of available block IDs.
        used_block_ids (set[int]): Set of currently allocated block IDs.
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        Initialize the BlockManager with a specific number of blocks of given size.
        
        Args:
            num_blocks (int): Total number of blocks to manage.
            block_size (int): Number of tokens per block.
        """
        if num_blocks <= 0:
            raise ValueError(f"Number of blocks must be positive, got {num_blocks}")
            
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def _flatten_token_ids(token_ids):
        """
        Flatten a potentially nested list of token IDs into a simple list.
        
        Args:
            token_ids: Potentially nested list/tuple/array of token IDs.
            
        Returns:
            list: Flattened list of token IDs.
        """
        flat = []
        for item in token_ids:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(BlockManager._flatten_token_ids(item))
            else:
                flat.append(item)
        return flat

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        Compute a hash value for a sequence of token IDs, optionally with a prefix hash.
        
        Args:
            token_ids: List of token IDs to hash.
            prefix: Optional prefix hash value to include (-1 if no prefix).
            
        Returns:
            int: A 64-bit hash value for the token sequence.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
            
        token_ids = cls._flatten_token_ids(token_ids)
        arr = np.array(token_ids, dtype=np.int32)
        h.update(arr.tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        Allocate a specific block, marking it as used.
        
        Args:
            block_id: ID of the block to allocate.
            
        Returns:
            Block: The allocated block.
            
        Raises:
            ValueError: If the block is already in use (ref_count > 0).
        """
        block = self.blocks[block_id]
        if block.ref_count > 0:
            raise ValueError(f"Block {block_id} is already in use (ref_count={block.ref_count})")
            
        block.reset()
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        Deallocate a specific block, marking it as free.
        
        Args:
            block_id: ID of the block to deallocate.
            
        Returns:
            Block: The deallocated block.
            
        Raises:
            ValueError: If the block is still in use (ref_count > 0).
        """
        block = self.blocks[block_id]
        if block.ref_count > 0:
            raise ValueError(f"Cannot deallocate block {block_id} with ref_count={block.ref_count}")
            
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        return block

    def can_allocate(self, seq: Sequence) -> bool:
        """
        Check if there are enough free blocks to allocate for a given sequence.
        
        Args:
            seq: The sequence to check allocation for.
            
        Returns:
            bool: True if there are enough free blocks, False otherwise.
        """
        num_blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_block_ids) >= num_blocks_needed

    def allocate(self, seq: Sequence) -> bool:
        """
        Allocate blocks for a sequence, reusing cached blocks when possible.
        """
        # Clear any existing block table
        seq.block_table = []
        
        h = -1
        cache_miss = False
        blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        
        for i in range(blocks_needed):
            token_ids = seq.block(i) if i < seq.num_blocks else []
            # Only compute hash if we have a full block
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # Check if we have a cache hit (matching hash and token IDs)
            if block_id == -1 or h == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
                
            if cache_miss:
                # Need to allocate a new block
                if not self.free_block_ids:
                    raise RuntimeError("No free blocks available for allocation")
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit - reuse existing block
                seq.num_cached_tokens += self.block_size
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
                    
            # Update the block with new hash and token IDs if needed
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                
            seq.block_table.append(block_id)

        return True

    def deallocate(self, seq: Sequence) -> None:
        """
        Deallocate all blocks associated with a sequence.
        
        Decrements the reference count of each block in the sequence's block table,
        and fully deallocates blocks whose reference count reaches zero.
        
        Args:
            seq: The sequence whose blocks should be deallocated.
        """
        if not hasattr(seq, 'block_table') or not seq.block_table:
            # Nothing to deallocate
            return
            
        for block_id in reversed(seq.block_table):
            if block_id >= len(self.blocks) or block_id < 0:
                # Skip invalid block IDs
                continue
                
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                try:
                    self._deallocate_block(block_id)
                except ValueError:
                    # If deallocation fails, just add the block back to free_block_ids
                    if block_id not in self.free_block_ids:
                        self.free_block_ids.append(block_id)
                    if block_id in self.used_block_ids:
                        self.used_block_ids.remove(block_id)
                        
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        Check if a new token can be appended to the sequence.
        
        A new block is needed if the sequence length modulo block_size is 1 
        (i.e., we've just filled a block and need a new one for the next token).
        
        Args:
            seq: The sequence to check.
            
        Returns:
            bool: True if a new token can be appended, False otherwise.
        """
        # We need a new block only when we've just filled a block completely
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 0)

    def may_append(self, seq: Sequence) -> None:
        """
        Manage block allocation when a new token is appended to a sequence.
        
        This function handles three cases based on the sequence length relative to block_size:
        
        1. When len(seq) % block_size == 1: 
           We've just filled a block completely and need to allocate a new block for the next token.
           
        2. When len(seq) % block_size == 0: 
           The current token is the last one in a block, so we compute and set the hash for this now-complete block.
           
        3. Otherwise: 
           We're still filling an already allocated block, no special action needed.
        
        Args:
            seq: The sequence that has had a token appended.
            
        Raises:
            ValueError: If the sequence has an empty block table.
            RuntimeError: If a new block is needed but none are available.
        """
        # Validate sequence has a block table
        if not hasattr(seq, 'block_table') or not seq.block_table:
            raise ValueError(f"Empty block table for sequence {seq.seq_id}")
            
        # Get the last block in the sequence's block table
        last_block_id = seq.block_table[-1]
        if last_block_id >= len(self.blocks) or last_block_id < 0:
            raise ValueError(f"Invalid block ID {last_block_id} for sequence {seq.seq_id}")
            
        last_block = self.blocks[last_block_id]
        seq_len = len(seq)
        
        # Case 1: Need to allocate a new block (first token of a new block)
        if seq_len % self.block_size == 1 and seq_len > 1:
            # We need a new block
            if not self.free_block_ids:
                raise RuntimeError(f"No free blocks available to append to sequence {seq.seq_id}")
                
            block_id = self.free_block_ids[0]
            try:
                self._allocate_block(block_id)
                seq.block_table.append(block_id)
            except ValueError as e:
                # If allocation fails, try to recover by finding another free block
                if len(self.free_block_ids) > 1:
                    self.free_block_ids.popleft()  # Remove the problematic block
                    if not self.free_block_ids:
                        raise RuntimeError(f"No free blocks available after allocation error: {str(e)}")
                    block_id = self.free_block_ids[0]
                    self._allocate_block(block_id)
                    seq.block_table.append(block_id)
                else:
                    raise RuntimeError(f"Block allocation failed: {str(e)}")
            
        # Case 2: Just filled a block completely
        elif seq_len % self.block_size == 0 and seq_len > 0:
            # Compute hash for the now-complete block
            try:
                token_ids = seq.block(seq.num_blocks-1)
                # Get the hash of the previous block (if any) as prefix
                prefix = -1
                if len(seq.block_table) > 1:
                    prev_block_id = seq.block_table[-2]
                    if 0 <= prev_block_id < len(self.blocks):
                        prefix = self.blocks[prev_block_id].hash
                
                # Compute and store the hash for the current block
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            except Exception as e:
                # If hash computation fails, we'll still keep the block but without a valid hash
                # This prevents the block from being reused via cache hits, but doesn't break generation
                last_block.hash = -1
                # Log this situation but don't halt execution
                print(f"Warning: Failed to compute hash for block {last_block_id}: {str(e)}")
            
        # Case 3: Still filling an already allocated block
        else:
            # We're in the middle of filling a block, ensure hash is -1 as expected
            if last_block.hash != -1:
                # This is unexpected but might happen - reset it to the expected state
                last_block.hash = -1
