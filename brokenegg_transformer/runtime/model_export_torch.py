# Copyright 2020 Katsuya Iida.

from brokenegg_transformer.runtime import transformer_torch
import torch
import argparse

def export(model_file, output_file):
    model = transformer_torch.load_model(model_file)
    model.eval()
    inputs = torch.tensor([[1,2,3]], dtype=torch.long)
    targets = torch.tensor([[1,2,3,4,5]], dtype=torch.long)

    torch.onnx.export(model,
                      (inputs, targets),
                      output_file,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names = ['inputs', 'targets'],
                      output_names = ['outputs'],
                      dynamic_axes={'inputs' : {0 : 'batch_size', 1: 'input_lengths'},
                                    'targets' : {0 : 'batch_size', 1: 'target_lengths'},
                                    'outputs' : {0 : 'batch_size', 1: 'target_lengths'}})

def export_infer(model_file, output_file):
    model = transformer_torch.load_model2(model_file)
    model.eval()
    inputs = torch.tensor([[1]], dtype=torch.long)
    targets = torch.tensor([[1]], dtype=torch.long)
    outputs = model(inputs, targets)
    script_model = torch.jit.script(model)
    torch.onnx.export(script_model,
                    (inputs, targets),
                    output_file,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names = ['inputs', 'targets'],
                    output_names = ['outputs'],
                    dynamic_axes={'inputs' : {0 : 'batch_size', 1: 'input_lengths'},
                                'targets' : {0 : 'batch_size', 1: 'target_lengths'},
                                'outputs' : {0 : 'batch_size', 1: 'output_lengths'}},
                    example_outputs=outputs)

def main():
    parser = argparse.ArgumentParser('Export PyTorch model to ONNX format.')
    parser.add_argument('--weight',
        default='export/brokenegg-20200711.npz', type=str,
        help='The NumPy weight file.')
    parser.add_argument('--output',
        default="export/brokenegg-20200711_torch_2.onnx", type=str,
        help='The output file.')
    args = parser.parse_args()
    export_infer(args.weight, args.output)

if __name__ == '__main__':
    main()