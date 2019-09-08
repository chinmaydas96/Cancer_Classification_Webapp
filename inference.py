import json
import torch
import torch.nn.functional as F

from commons import get_model, transform_image

model = get_model()
class_dict = {1:'Malignent',0:'Benign'}

def get_prediction(image_bytes):
	transform_image(image_bytes=image_bytes)
	#try:
	tensor = transform_image(image_bytes=image_bytes)
	model.eval()
	with torch.no_grad():
		output = model.forward(tensor)
	#_, predicted = torch.max(output.data, 1)   
	probability = F.softmax(output.data,dim=1)        
	prob,clas = probability.topk(1)
    #outputs = model.forward(tensor)
	#print(prob,clas)

	#except Exception:
	#    return 0, 'error'
	#_, y_hat = outputs.max(1)
	#predicted_idx = str(y_hat.item())
	#print(clas)
	return class_dict[clas[0][0].tolist()],round(prob[0][0].tolist(),3)
