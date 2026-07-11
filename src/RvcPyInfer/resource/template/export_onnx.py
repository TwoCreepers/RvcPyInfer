# ----------------------------------------------------------------
# 这只是一个用于自动生成导出脚本的模板而已，它不是正常代码！
# ----------------------------------------------------------------

# -- Template --
ModelPath = None
ExportedPath = None
ParsePrefix = None
Root = None

ModelPath = '/./..{code.InsertionPoint.0}.././'
ExportedPath = '/./..{code.InsertionPoint.1}.././'
ParsePrefix = '/./..{code.InsertionPoint.2}.././'
Root = '/./..{code.InsertionPoint.3}.././'

assert ModelPath is not None, "ModelPath 不能为 None"
assert ExportedPath is not None, "ExportedPath 不能为 None"
assert ParsePrefix is not None, "ParsePrefix 不能为 None"

# 以上是需要动态生成的
device = "cpu" # 一般 cpu 就可以

import sys
if Root is not None:
    sys.path.insert(0, Root)

def set_hook():
    import infer.lib.infer_pack.attentions as attentions # pyright: ignore[reportMissingImports]

    def attention_fix(
        self,
        query: attentions.torch.Tensor,
        key: attentions.torch.Tensor,
        value: attentions.torch.Tensor,
        mask: attentions.Optional[attentions.torch.Tensor] = None,
    ):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = attentions.torch.matmul(query / attentions.math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            # source: assert (
            # source:     t_s == t_t
            # source: ), "Relative attention is only available for self-attention."
            if not attentions.torch.jit.is_tracing(): # 大概率是没有问题的，大不了咱直接抛维度错误
                attentions.torch._assert(
                    t_s == t_t,
                    "Relative attention is only available for self-attention."
                )
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / attentions.math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            # source: assert t_s == t_t, "Proximal bias is only available for self-attention."
            if not attentions.torch.jit.is_tracing():
                attentions.torch._assert(
                    t_s == t_t,
                    "Proximal bias is only available for self-attention."
                )
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                # source: assert (
                # source:    t_s == t_t
                # source: ), "Local attention is only available for self-attention."
                if not attentions.torch.jit.is_tracing():
                    attentions.torch._assert(
                        t_s == t_t,
                        "Local attention is only available for self-attention."
                    )
                block_mask = (
                    attentions.torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = attentions.F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = attentions.torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn
    attentions.MultiHeadAttention.attention = attention_fix

    def _get_relative_embeddings_fix(self, relative_embeddings, length: int):
        max_relative_position = 2 * self.window_size + 1 # 不知道有什么用，反正不敢删
        # Pad first before slice to avoid using cond ops.
        pad_length: int = attentions.torch.clamp(length - (self.window_size + 1), min=0)
        slice_start_position = attentions.torch.clamp((self.window_size + 1) - length, min=0)
        slice_end_position = slice_start_position + 2 * length - 1
        # source: if pad_length > 0:
        if True: # pad 0 那不就是没有 pad嘛，搞个分支干什么，我还得改
            padded_relative_embeddings = attentions.F.pad(
                relative_embeddings,
                # commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
                [0, 0, pad_length, pad_length, 0, 0],
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings
    attentions.MultiHeadAttention._get_relative_embeddings = _get_relative_embeddings_fix

    def _relative_position_to_absolute_position_fix(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = attentions.F.pad(
            x,
            #   commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]])
            [0, 1, 0, 0, 0, 0, 0, 0],
        )

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = attentions.F.pad(
            x_flat,
            # commons.convert_pad_shape([[0, 0], [0, 0], [0, int(length) - 1]])
            # source: [0, int(length) - 1, 0, 0, 0, 0],
            [0, length - 1, 0, 0, 0, 0],
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final
    attentions.MultiHeadAttention._relative_position_to_absolute_position = _relative_position_to_absolute_position_fix

    def _absolute_position_to_relative_position_fix(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = attentions.F.pad(
            x,
            # commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, int(length) - 1]])
            # source: [0, int(length) - 1, 0, 0, 0, 0, 0, 0],
            [0, length - 1, 0, 0, 0, 0, 0, 0],
        )
        # source: x_flat = x.view([batch, heads, int(length**2) + int(length * (length - 1))])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = attentions.F.pad(
            x_flat,
            #    commons.convert_pad_shape([[0, 0], [0, 0], [int(length), 0]])
            [length, 0, 0, 0, 0, 0],
        )
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final
    
    attentions.MultiHeadAttention._absolute_position_to_relative_position = _absolute_position_to_relative_position_fix

def main():
    import torch # pyright: ignore[reportMissingImports]

    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    print(f'{ParsePrefix}/./..{{"sr":{cpt["config"][-1]}}}.././')
    version = cpt["version"]
    hidden_channels = 256 if version == "v1" else 768

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200)  # nsf基频
    test_ds = torch.LongTensor([0])  # 说话人ID
    test_rnd = torch.rand(1, 192, 200)  # 噪声（加入随机因子）

    set_hook() # 强制误差调整大法！强制替换不支持追踪的实现

    from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM # pyright: ignore[reportMissingImports]

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], version=version, is_half=False
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) 多角色混合轨道导出

    try: 
        torch.onnx.export(
            net_g,
            (
                test_phone.to(device),
                test_phone_lengths.to(device),
                test_pitch.to(device),
                test_pitchf.to(device),
                test_ds.to(device),
                test_rnd.to(device),
            ),
            ExportedPath,
            dynamic_axes={ # 记得批处理维度！
                "phone": {0: "batch_size", 1: "frame_sequence"}, 
                "pitch": {0: "batch_size", 1: "frame_sequence"},
                "pitchf": {0: "batch_size", 1: "frame_sequence"},
                "rnd": {0: "batch_size", 2: "frame_sequence"},
                "audio": {0: "batch_size", 2: "sampling_sequence"}
            },
            do_constant_folding=False,
            opset_version=16, # 其实我觉得可以更高
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
    except Exception as e:
        old_line = "\n"
        new_line = "\\n" # python 的奇妙切词
        old_str = "\""
        new_str = "\\\""
        print(f'{ParsePrefix}/./..{{"status":"fail","msg":"{str(e).replace(old_line, new_line).replace(old_str, new_str)}"}}.././')
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f'{ParsePrefix}/./..{{"status":"success"}}.././')
    sys.exit(0)

if __name__ == "__main__":
    main()