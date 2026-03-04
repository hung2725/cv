import os
label_path = r'D:\Hoc_Tap\HK252\Thị Giác Máy Tính\CK\data\traffic_light_data\valid\labels'
for filename in os.listdir(label_path):
    with open(os.path.join(label_path, filename), 'r') as f:
        for line in f:
            if len(line.split()) > 5:
                print(f"File bị lỗi định dạng: {filename}")