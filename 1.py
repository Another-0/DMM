import torch

# 加载权重文件
checkpoint_path = "weights/dronevehicle_best.pth"
state_dict = torch.load(checkpoint_path)

# 创建一个新的 state_dict
new_state_dict = {}


# 修改模块名称
for key, value in state_dict["state_dict"].items():
    if "illuBlock" in key:
        print(key)
        new_key = key.replace("illuBlock", "mtablock")
    else:
        new_key = key
        # print(new_key)

    new_state_dict[new_key] = value

# 保存修改后的权重

new_checkpoint_path = "weights/dronevehicle_best_new.pth"
new_pth = {"meta": state_dict["meta"], "state_dict": new_state_dict}
torch.save(new_pth, new_checkpoint_path)

print(f"权重文件已保存至 {new_checkpoint_path}")
