import torch
import torch.distributed as dist


def broadcast_value(value, rank):
    """
    Broadcast a string value from rank 0 to all other ranks.

    Args:
        value: The string value to broadcast (only rank 0 needs to provide this).
        rank: The rank of the current process.

    Returns:
        The broadcasted string value for all ranks.
    """
    device = torch.cuda.current_device()

    if rank == 0:
        value_bytes = b"" if value is None else value.encode("utf-8")
        length_tensor = torch.tensor(len(value_bytes), dtype=torch.long, device=device)
        dist.broadcast(length_tensor, src=0)

        if len(value_bytes) > 0:
            value_tensor = torch.frombuffer(
                bytearray(value_bytes), dtype=torch.uint8
            ).to(device)
            dist.broadcast(value_tensor, src=0)
            return value
        else:
            return None
    else:
        length_tensor = torch.tensor(0, dtype=torch.long, device=device)
        dist.broadcast(length_tensor, src=0)

        if length_tensor.item() > 0:
            value_tensor = torch.zeros(
                length_tensor.item(), dtype=torch.uint8, device=device
            )
            dist.broadcast(value_tensor, src=0)
            return value_tensor.cpu().numpy().tobytes().decode("utf-8")
        else:
            return None
