from transformers import AutoConfig
import torch

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

def HoloV(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 0.09
    pruned_image_tokens_list = []

    for b in range(B):
        image_token = image_tokens[b]  # [N, D]
        image_attention = attention[b]  # [N]

        # Calculate dynamic patch size - handle uneven divisions
        patch_size = N // num_patches
        remainder = N % num_patches

        # Create patches with potentially uneven sizes
        image_tokens_patches = []
        attention_patches = []
        start_idx = 0

        for p in range(num_patches):
            # Last few patches get an extra token if there's a remainder
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size

            if current_patch_size > 0:  # Skip empty patches
                image_tokens_patches.append(image_token[start_idx:end_idx])
                attention_patches.append(image_attention[start_idx:end_idx])

            start_idx = end_idx

        # Process each patch separately
        patch_scores = []
        all_patches = []

        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]  # [current_patch_size, D]
            patch_attn = attention_patches[p]  # [current_patch_size]
            current_patch_size = len(patch_tokens)

            if current_patch_size <= 1:
                # If patch has only one token or is empty, handle specially
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                continue

            with torch.no_grad():
                # Normalize patch tokens
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp)

                # Compute similarity matrix
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1))

                # Create eye mask of appropriate size
                eye_mask = 1 - torch.eye(current_patch_size, device=device)
                S_masked = S * eye_mask

                # Compute mean and variance
                valid_entries = current_patch_size - 1
                mean_sim = S_masked.sum(dim=1) / valid_entries
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries

                # Scale attention
                patch_attn_scaled = patch_attn * 1e3

                # Scale variance
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) / 
                              (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                # Calculate token scores
                token_scores =  patch_attn_scaled + alpha * var_sim_scaled

                # Compute patch score
                patch_score = token_scores.mean()
                patch_scores.append(patch_score)
                all_patches.append(patch_tokens)

        # Convert to tensor
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)

        # Allocate new tokens based on scores
        if len(patch_scores) > 0:
            weights = (patch_scores ) / ((patch_scores).sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            # Distribute remaining tokens
            remaining = new_image_token_num - allocated.sum()
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            # Handle token overflow
            new_patches = []
            for i, (patch, alloc) in enumerate(zip(all_patches, allocated)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    # Keep all tokens in this patch
                    new_patches.append(patch)
                else:
                    # Sample tokens based on attention scores
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])

            # Combine all selected tokens
            if new_patches:
                new_image_tokens = torch.cat(new_patches, dim=0)
            else:
                new_image_tokens = torch.zeros((0, D), device=device)
        else:
            # No patches to process
            new_image_tokens = torch.zeros((0, D), device=device)

        # Pad or truncate to match expected new_image_token_num
        actual_tokens = new_image_tokens.size(0)
        if actual_tokens < new_image_token_num:
            # Pad with zeros if we don't have enough tokens
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
        elif actual_tokens > new_image_token_num:
            # Truncate if we have too many tokens
            new_image_tokens = new_image_tokens[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)

    # Stack batches
    return torch.stack(pruned_image_tokens_list, dim=0).to(image_tokens.dtype) 