import torch
print(torch.cuda.is_available())  # True nếu CUDA hoạt động
print(torch.cuda.device_count())  # Số lượng GPU khả dụng
print(torch.cuda.get_device_name(0))  # Tên GPU đầu tiên
