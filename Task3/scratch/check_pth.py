import torch
import os

path = r'c:\University\Eight\GenAI\Assignments\22F-3396_22F-3369_Assignment03\Task3\cyclegan_final.pth'
if os.path.exists(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        print("Keys in checkpoint:", checkpoint.keys())
    else:
        print("Checkpoint is not a dict, type:", type(checkpoint))
else:
    print("File not found")
