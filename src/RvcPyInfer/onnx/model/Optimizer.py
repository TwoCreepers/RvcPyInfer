from collections import Counter

from ...type_alist import PathLike


class Optimizer:
    def __init__(self, model: PathLike) -> None:
        import onnx  # pyright: ignore[reportMissingImports]
        self.model = onnx.load_model(model)

    def simplify(self, output: PathLike, is_static_batch: bool = True, is_print_result: bool = True) -> None:
        orig_total, orig_types = self.get_node_stats(self.model)

        if is_static_batch:
            for inp in self.model.graph.input:
                tensor_type = inp.type.tensor_type
                if tensor_type.shape.dim[0].dim_param:
                    tensor_type.shape.dim[0].dim_value = 1

        # 不得已直接调用 C 导出，它的 python 包装内部没有处理我还有可能保留一点动态维度的情况
        model_bytes = self.model.SerializeToString()
        skip_constant_folding = False # 我不管了
        skip_shape_inference = False
        import onnxsim.onnxsim_cpp2py_export as C  # pyright: ignore[reportMissingImports]
        model_opt_bytes = C.simplify(
            model_bytes,
            [],
            not skip_constant_folding,
            not skip_shape_inference,
            1024 * 1024 * 1024, # 这边导出的模型基本就 100M，随便写写了
        )
        import onnx  # pyright: ignore[reportMissingImports]
        model_simp = onnx.load_from_string(model_opt_bytes)
        onnx.save_model(model_simp, output)
        print(f"模型已保存至: {output}")

        simp_total, simp_types = self.get_node_stats(model_simp)

        if is_print_result:
            self.print_result(orig_total, orig_types, simp_total, simp_types)

    @staticmethod
    def get_node_stats(model):
        """获取模型的总算子数和按类型分类的算子数"""
        total_nodes = len(model.graph.node)
        # 统计每种算子类型的出现次数
        node_types = [node.op_type for node in model.graph.node]
        type_counts = Counter(node_types)
        return total_nodes, type_counts
    
    @staticmethod
    def print_result(orig_total, orig_types, simp_total, simp_types):
        diff = orig_total - simp_total
        print("="*40)
        print(f"减少了 {diff} 个算子，约占原始模型的 {diff / orig_total * 100.0:.1f} %")
        print("="*40)

        print("\n--- 算子类型变化明细 ---")
        all_types = set(list(orig_types.keys()) + list(simp_types.keys()))

        # 按减少的数量降序排列
        type_diffs = []
        for t in all_types:
            orig_count = orig_types.get(t, 0)
            simp_count = simp_types.get(t, 0)
            if orig_count != simp_count:
                type_diffs.append((t, orig_count, simp_count, orig_count - simp_count))

        # 排序：减少最多的排在前面
        type_diffs.sort(key=lambda x: x[3], reverse=True)

        print(f"{'算子类型':<25} | {'优化前':>6} | {'优化后':>6} | {'变化量':>6}")
        print("-" * 55)
        for t, o, s, d in type_diffs:
            # d>0 表示算子减少，d<0 表示算子增加
            sign = "↓" if d > 0 else "↑"
            print(f"{t:<25} | {o:>6} | {s:>6} | {sign} {abs(d):>4}")