import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Memuat model dari file pickle
with open('trained_models.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Akses model dari dictionary
model = model_data['model']

# Tentukan tipe input
input_dimension = model.n_features_in_
initial_type = [('float_input', FloatTensorType([None, input_dimension]))]

# Konversi model ke format ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Simpan model ONNX ke file
with open("trained_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
