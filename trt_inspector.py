import tensorrt as trt

# Load the TensorRT engine from file
engine_path = "fp16-effinetb1-validation.engine"
with open(engine_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
engine = runtime.deserialize_cuda_engine(engine_data)

# Iterate over the layers and inspect precision
for idx in range(engine.num_layers):
    layer = engine.get_binding_name(idx)
    precision = engine.get_binding_dtype(idx)
    print(f"Layer: {layer}, Precision: {precision}")

# Clean up
del engine
