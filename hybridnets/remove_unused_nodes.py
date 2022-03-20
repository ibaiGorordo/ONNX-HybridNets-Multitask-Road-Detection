import onnx
import onnxoptimizer
import argparse

def optimized_model(model_path):

	onnx_model = onnx.load(model_path)
	passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
	optimized_model = onnxoptimizer.optimize(onnx_model, passes)

	onnx.save(optimized_model, model_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Optimize the ONNX model to remove unused nodes')
	parser.add_argument('path', help='path to the model')
	args = parser.parse_args()

	optimized_model(args.path)
