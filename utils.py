import torch
import math

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 确保 attn_bias 的维度与 attn_mask 匹配
            if attn_mask.dim() > attn_bias.dim():
                # 如果 attn_mask 维度更高，需要扩展 attn_bias
                attn_bias = attn_bias.unsqueeze(0).expand_as(attn_mask)
            elif attn_mask.dim() < attn_bias.dim():
                # 如果 attn_mask 维度更低，需要压缩 attn_mask
                attn_mask = attn_mask.unsqueeze(0)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_logits = attn_weight
    return attn_weight @ value,attn_logits





def HoloV(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    attention = attention.unsqueeze(0)
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 1
    beta = 0.09
    power = 1
    pruned_image_tokens_list = []
    final_positions_list = []  

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
        patch_start_indices = []  

        for p in range(num_patches):
            # Last few patches get an extra token if there's a remainder
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size

            if current_patch_size > 0:  # Skip empty patches
                image_tokens_patches.append(image_token[start_idx:end_idx])
                attention_patches.append(image_attention[start_idx:end_idx])
                patch_start_indices.append(start_idx)  # 记录patch起始位置

            start_idx = end_idx

        # Process each patch separately
        patch_scores = []
        all_patches = []
        all_patch_indices = []  

        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]  # [current_patch_size, D]
            patch_attn = attention_patches[p]  # [current_patch_size]
            patch_start = patch_start_indices[p]  
            current_patch_size = len(patch_tokens)

            
            patch_indices = torch.arange(patch_start, patch_start + current_patch_size, device=device)

            if current_patch_size <= 1:
                # If patch has only one token or is empty, handle specially
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)
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
                token_scores = alpha * patch_attn_scaled + beta * var_sim_scaled

                # Compute patch score
                patch_score = token_scores.mean()
                patch_scores.append(patch_score)
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)

        # Convert to tensor
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)

        # Allocate new tokens based on scores
        if len(patch_scores) > 0:
            weights = (patch_scores ** power) / ((patch_scores ** power).sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            # Distribute remaining tokens
            remaining = new_image_token_num - allocated.sum()
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            # Handle token overflow
            new_patches = []
            final_positions = []  

            for i, (patch, alloc, patch_indices) in enumerate(zip(all_patches, allocated, all_patch_indices)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    # Keep all tokens in this patch
                    new_patches.append(patch)
                    final_positions.append(patch_indices)
                else:
                    # Sample tokens based on attention scores
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])
                    final_positions.append(patch_indices[top_indices])

            # Combine all selected tokens
            if new_patches:
                new_image_tokens = torch.cat(new_patches, dim=0)
                final_positions = torch.cat(final_positions, dim=0) 
            else:
                new_image_tokens = torch.zeros((0, D), device=device)
                final_positions = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # No patches to process
            new_image_tokens = torch.zeros((0, D), device=device)
            final_positions = torch.zeros(0, dtype=torch.long, device=device)

        # Pad or truncate to match expected new_image_token_num
        actual_tokens = new_image_tokens.size(0)
        if actual_tokens < new_image_token_num:
            # Pad with zeros if we don't have enough tokens
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
            
            
            padding_positions = torch.full((new_image_token_num - actual_tokens,), -1, dtype=torch.long, device=device)
            final_positions = torch.cat([final_positions, padding_positions], dim=0)
        elif actual_tokens > new_image_token_num:
            # Truncate if we have too many tokens
            new_image_tokens = new_image_tokens[:new_image_token_num]
            final_positions = final_positions[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)
        final_positions_list.append(final_positions)

    
    return torch.stack(pruned_image_tokens_list, dim=0), torch.stack(final_positions_list, dim=0).squeeze(0)







def adjust_ids(input_ids, position_ids, image_token_start, image_token_len, final_position):
    
    if input_ids is not None:
        pre_ids = input_ids[:, :image_token_start]
        post_ids = input_ids[:, image_token_start + image_token_len:]
        kept_ids = input_ids[:, image_token_start:image_token_start + image_token_len]
        kept_ids = kept_ids[:, final_position]  
        adjusted_input_ids = torch.cat([pre_ids, kept_ids, post_ids], dim=1)
    else:
        adjusted_input_ids = None
    
    if position_ids is not None:
        pre_pos = position_ids[:, :, :image_token_start]
        post_pos = position_ids[:, :, image_token_start + image_token_len:]
        kept_pos = position_ids[:, :, image_token_start:image_token_start + image_token_len]
        kept_pos = kept_pos[:, :, final_position]  
        adjusted_position_ids = torch.cat([pre_pos, kept_pos, post_pos], dim=2)
    else:
        adjusted_position_ids = None
        
    return adjusted_input_ids, adjusted_position_ids