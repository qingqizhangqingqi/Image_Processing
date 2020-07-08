from mmdet.nvidia import pynvml
import time

def get_free_gpu(need_memory):
  count = 0
  pynvml.nvmlInit()
  device_count = pynvml.nvmlDeviceGetCount()
  while True:
    for i in range(device_count):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      name = pynvml.nvmlDeviceGetName(handle)
      meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
      free_memory = meminfo.free/1024/1024/1024
      if free_memory > need_memory:
        print("find free gpu, gpu id = {}, free memory = {:.2f}".format(i, free_memory))
        return str(i)

      print("id = {}, gpu name = {}, free memory = {:.2f}GB".format(i, name, free_memory))

    time.sleep(2)
    count += 1
    print("find free gpu, try count = {}".format(count))

if __name__ == "__main__":
	get_free_gpu(5)
