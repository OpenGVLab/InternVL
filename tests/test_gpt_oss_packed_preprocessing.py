import ast
import unittest
from pathlib import Path


TRAINING_SCRIPT = (
    Path(__file__).parents[1]
    / 'internvl_chat_gpt_oss'
    / 'internvl'
    / 'train'
    / 'internvl_chat_finetune.py'
)


class PackedPreprocessingTests(unittest.TestCase):
    def test_all_supervised_modalities_forward_packed_dataset_mode(self):
        tree = ast.parse(TRAINING_SCRIPT.read_text(encoding='utf-8'))
        dataset_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'LazySupervisedDataset'
        )
        methods = {
            node.name: node
            for node in dataset_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for method_name in (
            'multi_modal_get_item',
            'multi_modal_multi_image_get_item',
            'video_get_item',
            'pure_text_get_item',
        ):
            with self.subTest(method=method_name):
                preprocess_calls = [
                    node
                    for node in ast.walk(methods[method_name])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'preprocess_function'
                ]
                self.assertEqual(len(preprocess_calls), 1)

                packed_keyword = next(
                    (keyword for keyword in preprocess_calls[0].keywords if keyword.arg == 'use_packed_ds'),
                    None,
                )
                self.assertIsNotNone(packed_keyword)
                self.assertEqual(
                    ast.dump(packed_keyword.value),
                    ast.dump(
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='use_packed_ds',
                            ctx=ast.Load(),
                        )
                    ),
                )


if __name__ == '__main__':
    unittest.main()
